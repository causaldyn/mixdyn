"""An arm of this engine: one fit of what an arm reads of a world, and the plans made from it.

An arm reads one export (`causaldyn_bench.scorecard.observe.Observation`) and nothing else, and an
arm that reads no lift test is handed the export without them (`withheld`). An arm that does not
plan a geo export's cells is handed its market (`scorecard.arms.panel.market`) and plans one weekly
spend a channel, which the bench allots to the cells; its record says so (`plan_level`, `market`
where it planned the market and `cells` where it planned each cell, on a geo export alone). It
fits once and plans once a variant: a plug-in, a posterior's expectation, a tail's, are one fit's
cost and several arms' plans. `records` runs an arm on an observation and returns each variant's
record as the bench's scorer reads a tool's (`causaldyn_bench.scorecard.scoring`): the plan and
the observation's digest, and beside them what the variant claims, the fit's response where the
variant records it, how the fit went, its settings, the versions it ran on and its cost, the fit's
and the variant's own. Beside the arm's flags, a plan of one weekly spend a channel records how
far its spend lies outside each channel's logged range (`outside`). An arm whose fit raises plays
the status quo, as the bench scores every arm whose fit failed: each variant records the error and
its traceback. A variant whose plan raises records its own, and the others stand. A failure's kind
is recorded beside it (`failure_kind`): the fit raised, the plan raised, or the plan was not one a
record holds.

The cost is each step's wall and CPU seconds and the seconds JAX spent compiling within them
(`COMPILES`), which a fresh process a world pays in full; the process's peak resident set
(`VmHWM`) once the fit returns and once the last variant is planned, the world's; where a
variant's plan reads a Markov chain's draws, their least bulk effective sample size per CPU second
of the fit; and where the process ran, the CPU's model and the threads it was given, as the
bench's tool arms record them for its compute table (`causaldyn_bench.scorecard.report`), with
the processes XLA's CPU client was told it has beside them.
"""

from __future__ import annotations

import dataclasses
import importlib.metadata
import math
import os
import platform
import resource
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from causaldyn_bench.budget_regret import lift_rows
from jax import monitoring

from mixdyn.inference.diagnostics import out_of_range
from scorecard.arms.panel import market, second

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Mapping

    from causaldyn_bench.scorecard.observe import Observation

PACKAGES = ("mixdyn", "causal-hybrid-control", "jax", "numpy", "scipy")
"""The packages a record names the versions of."""
COMPILES = (
    "/jax/core/compile/jaxpr_trace_duration",
    "/jax/core/compile/jaxpr_to_mlir_module_duration",
    "/jax/core/compile/backend_compile_duration",
)
"""JAX's events for tracing a function, lowering it and XLA's compiling it, each once a process."""


@dataclass(frozen=True, slots=True)
class Planned:
    """One variant's plan for the quarter and what it claims of it.

    `weekly` is each cell's weekly spend. `gain` is the plan's return over the status quo's, in the
    outcome's units over the quarter and the kernel's tail, and `gain_interval` its central
    interval, where the variant reads one; `forecast` is draws of the quarter's weekly sales at the
    status quo, `(draws, weeks)`; `response` is the fit's draws of each channel and its account of
    the history, as the bench maps a tool's (`scorecard.arms.mapped`). `flags` are the fit's
    diagnostics, `settings` what the variant was run with, and `absent` names what the variant
    does not give and why, so an absence is counted rather than imputed. `ess` is the least bulk
    effective sample size of any quantity of the Markov chains whose draws the plan reads, where
    it reads a chain's.
    """

    weekly: np.ndarray
    gain: float | None = None
    gain_interval: tuple[float, float] | None = None
    forecast: np.ndarray | None = None
    response: Mapping[str, Any] | None = None
    flags: Mapping[str, float | int | bool] = field(default_factory=dict)
    settings: Mapping[str, Any] = field(default_factory=dict)
    absent: Mapping[str, str] = field(default_factory=dict)
    ess: float | None = None


@dataclass(frozen=True, slots=True)
class Spent:
    """A step's wall and CPU seconds, and JAX's seconds compiling within them."""

    wall: float
    cpu: float
    compile: float


class Arm[F](Protocol):
    """An arm: its variants' names, whether it reads the lift tests, one fit of an observation, and
    each variant's plan from it."""

    @property
    def variants(self) -> tuple[str, ...]:
        """The names its records are kept under, one a variant."""
        ...

    @property
    def tests(self) -> bool:
        """Whether it reads the export's lift tests; one that does not is handed none."""
        ...

    @property
    def cells(self) -> bool:
        """Whether it plans a geo export's every cell; one that does not is handed the export's
        market."""
        ...

    def fit(self, observation: Observation, *, smoke: bool) -> F:
        """The fit every variant plans from; reduced where `smoke`."""
        ...

    def plan(self, fit: F, variant: str, observation: Observation) -> Planned:
        """`variant`'s plan for `observation`'s quarter from `fit`."""
        ...


def withheld(observation: Observation) -> Observation:
    """`observation` without its lift tests, as at the ladder's first rung: what an arm that reads
    no test is handed, one observation at every rung of a world. A geo export's tests name their
    geos, so its rung without them names none; a version-4 export's universes and switchback test
    are tests too, and it is read as version 2 (`scorecard.arms.panel.second`)."""
    none = lift_rows({})
    if observation.population is not None:
        none = dataclasses.replace(none, geos=())
    return dataclasses.replace(second(observation), k=0, lift=none)


def records(arm: Arm[Any], observation: Observation, *, smoke: bool) -> dict[str, dict[str, Any]]:
    """Each of `arm`'s variants' record of `observation`, as the bench's scorer reads a tool's. An
    arm that reads no test fits and plans from the observation `withheld` gives, and one that does
    not plan a geo export's cells from its market."""
    base: dict[str, Any] = {
        "version": observation.version,
        "seed": observation.seed,
        "digest": observation.digest(),
        "family": observation.family,
        "environment": observation.environment,
        "k": observation.k,
        "python": platform.python_version(),
        "versions": {name: importlib.metadata.version(name) for name in PACKAGES},
    }
    if smoke:
        base["smoke"] = "smoke, not a result"
    seen = observation if arm.tests else withheld(observation)
    if observation.population is not None:
        base["plan_level"] = "cells" if arm.cells else "market"

    def fit() -> Any:
        """The arm's fit of what it reads: a market's reading is the fit's, and so its failure."""
        nonlocal seen
        if observation.population is not None and not arm.cells:
            seen = market(seen)
        return arm.fit(seen, smoke=smoke)

    out, costs = {}, {}
    with _compiling() as compiled:
        fitted, failure, fit_cost = _timed(fit, compiled)
        fit_peak = _peak_rss()
        for variant in arm.variants:
            if failure is not None:
                out[variant] = base | failure | {"failure_kind": "fit"}
                costs[variant] = _cost(fit_cost, None)
                continue
            planned, error, plan_cost = _timed(
                lambda v=variant: arm.plan(fitted, v, seen), compiled
            )
            costs[variant] = _cost(fit_cost, plan_cost)
            if error is not None:
                out[variant] = base | error | {"failure_kind": "plan"}
                continue
            try:
                out[variant] = base | _fields(planned, seen)
            except ValueError as malformed:
                out[variant] = base | {
                    "error": f"ValueError: {malformed}",
                    "traceback": traceback.format_exc(),
                    "failure_kind": "record",
                }
                continue
            ess = None if planned is None else planned.ess
            if ess is not None and math.isfinite(ess):
                costs[variant]["ess_bulk_per_fit_cpu_second"] = ess / fit_cost.cpu
    peak = {"fit_peak_rss_bytes": fit_peak, "peak_rss_bytes": _peak_rss()} | _machine()
    return {variant: record | {"cost": costs[variant] | peak} for variant, record in out.items()}


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


def _timed[T](
    work: Callable[[], T], compiled: Callable[[], float]
) -> tuple[T | None, dict[str, Any] | None, Spent]:
    """`work`'s value, or the error it raised and its traceback; its seconds."""
    began, spent, compiling = time.perf_counter(), _cpu(), compiled()
    try:
        value, failure = work(), None
    except Exception as error:  # an arm's fit and plans are a boundary: a failure is a result
        value = None
        failure = {"error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}
    return (
        value,
        failure,
        Spent(time.perf_counter() - began, _cpu() - spent, compiled() - compiling),
    )


def _cost(fit: Spent, plan: Spent | None) -> dict[str, float]:
    steps = [("fit", fit)] + ([("plan", plan)] if plan is not None else [])
    cost = {}
    for step, seconds in steps:
        cost |= {
            f"{step}_wall_seconds": seconds.wall,
            f"{step}_cpu_seconds": seconds.cpu,
            f"{step}_compile_seconds": seconds.compile,
        }
    cost["wall_seconds"] = sum(seconds.wall for _, seconds in steps)
    cost["cpu_seconds"] = sum(seconds.cpu for _, seconds in steps)
    cost["compile_seconds"] = sum(seconds.compile for _, seconds in steps)
    return cost


def _fields(plan: Planned | None, observation: Observation) -> dict[str, Any]:
    """The record's fields for `plan`.

    Raises:
        ValueError: on a plan that is not a finite weekly spend for each cell.
    """
    weekly = np.asarray(None if plan is None else plan.weekly, dtype=float)
    if (
        plan is None
        or weekly.shape != observation.status_quo.shape
        or not np.all(np.isfinite(weekly))
    ):
        msg = (
            f"a plan is a finite weekly spend for each of {observation.status_quo.size} cells,"
            f" not {weekly!r}"
        )
        raise ValueError(msg)
    absent = dict(plan.absent)
    record: dict[str, Any] = {
        "weekly": weekly.tolist(),
        # strict JSON holds no NaN: a flag that is not a number is null
        "flags": {
            k: None if _nan(v) else v
            for k, v in (dict(plan.flags) | outside(weekly, observation)).items()
        },
        "settings": dict(plan.settings),
        "error": None,
    }
    if plan.ess is not None and not math.isfinite(plan.ess):
        absent["ess"] = f"the least bulk ESS is not finite: {plan.ess}"
    claim = [plan.gain, *(plan.gain_interval or ())]
    if plan.gain is not None and any(_nan(v) for v in claim):
        absent["gain"] = f"the claimed gain is not finite: {claim}"
    elif plan.gain is not None:
        record["gain"] = float(plan.gain)
        if plan.gain_interval is not None:
            record["gain_interval"] = [float(v) for v in plan.gain_interval]
    if plan.forecast is not None:
        record["forecast"] = np.asarray(plan.forecast, dtype=float).tolist()
    response = None if plan.response is None else dict(plan.response)
    if response is not None and _finite(response):
        record["response"] = response
    elif response is not None:
        absent["response"] = "the response holds a number that is not finite"
    record["n/a"] = absent
    return record


def outside(weekly: np.ndarray, observation: Observation) -> dict[str, float]:
    """How far a plan's spend lies outside the range the history logged, as a share of its spend
    (`mixdyn.inference.diagnostics.out_of_range`), the plan's, `out of range`, and each channel's
    part of it, `<channel> out of range`: none for a plan of a geo export's cells, whose ranges
    are each cell's, nor for a plan that spends nothing or below 0 somewhere, which the flag does
    not turn into a failed record."""
    if (
        observation.population is not None
        or not float(np.sum(weekly)) > 0.0
        or bool(np.any(weekly < 0.0))
    ):
        return {}
    share, channels = out_of_range(observation.spend, weekly)
    return {"out of range": share} | {
        f"{name} out of range": float(part)
        for name, part in zip(observation.channels, channels, strict=True)
    }


def r2(y: np.ndarray, estimated: np.ndarray) -> float:
    """A fit's R² in sample: the share of the KPI's variance about its mean that the fit's
    estimate of it explains, `1 - sum (y - estimated)^2 / sum (y - mean y)^2`."""
    return float(1.0 - np.sum((y - estimated) ** 2) / np.sum((y - np.mean(y)) ** 2))


def _nan(value: object) -> bool:
    return isinstance(value, float) and not math.isfinite(value)


def _finite(value: object) -> bool:
    """Whether every number in `value`, a JSON document of dicts and lists, is finite."""
    if isinstance(value, dict):
        return all(_finite(v) for v in value.values())
    if isinstance(value, list):
        return all(_finite(v) for v in value)
    return not _nan(value)


def _cpu() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def _machine() -> dict[str, str | None]:
    """Where the process ran, as the bench's tool arms record it: the CPU's model as the kernel
    names it, None where it names none, and the threads the process was given, `OMP_NUM_THREADS`
    as the harness sets it (`scorecard.provenance.ONE_THREAD`), None where it is unset; and beside
    them `PJRT_NPROC`, which sizes XLA's CPU pools as `OMP_NUM_THREADS` does not, None where it is
    unset."""
    lines = Path("/proc/cpuinfo").read_text().splitlines()
    named = (line.split(":", 1)[1].strip() for line in lines if line.startswith("model name"))
    return {
        "cpu_model": next(named, None),
        "threads": os.environ.get("OMP_NUM_THREADS"),
        "pjrt_nproc": os.environ.get("PJRT_NPROC"),
    }


def _peak_rss() -> int:
    """The process's peak resident set so far, in bytes: `VmHWM`, which Linux keeps in kB.

    Raises:
        RuntimeError: where the process's status holds no `VmHWM`.
    """
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) * 1024
    msg = "/proc/self/status holds no VmHWM"
    raise RuntimeError(msg)
