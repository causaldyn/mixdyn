"""Fits and results saved to a file and loaded back, every return they give the same to the bit.

A file is a zip archive (ADR 0062). `manifest.json` names the format, its version, the versions of
mixdyn and causal-hybrid-control that wrote it, and the object as a tree of JSON nodes; each array
of the object is an `.npy` file of its own under `arrays/`, written and read with NumPy's
`allow_pickle=False`. Nothing in the archive is pickled, so loading a file runs no code of its own.

The tree spells each value by its kind: JSON's own for `None`, booleans, integers, strings and
finite floats, which JSON carries exactly; and an object with one `$`-key for the rest, a
non-finite float, a tuple, a list, a frozen set, a mapping (a `dict` or a `MappingProxyType`,
which a fit's fields are), a NumPy scalar, an array (a JAX array marked so that it loads as one,
in double precision as the engine runs), a date, an enum member, or a dataclass, its fields that
`__init__` takes by name. A dataclass or an enum is saved only when its class is in `REGISTRY`,
and a file naming a class the registry does not hold fails to load: the registry is the list of
what the format promises to read back. A dataclass is rebuilt by its own `__init__`, so a loaded
object passes the checks a fitted one passed.

`FORMAT_VERSION` moves when a file the previous version wrote would load as something else; a
file of another version is refused, never read as this one.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import enum
import io
import json
import math
import zipfile
from collections.abc import Mapping
from importlib import metadata
from pathlib import Path
from types import MappingProxyType
from typing import Any

import chc.allocation
import chc.dlm
import chc.lift
import jax
import jax.numpy as jnp
import numpy as np

from mixdyn.api import documents
from mixdyn.calibration import lift
from mixdyn.data import feedback, issues, panel, recurring
from mixdyn.decisions import flighting, geo_plan, posterior, scenarios
from mixdyn.decisions.plan_table import PlanTable, Spread
from mixdyn.decisions.returns import Costs, Returns, Undefined
from mixdyn.experiments import candidates, geo_design, reading, sdid, switchback
from mixdyn.experiments import value as experiment_value
from mixdyn.inference import diagnostics, nuts, sensitivity
from mixdyn.inference.loo import Loo, Psis
from mixdyn.models import (
    bayes_geo,
    bayes_mmm,
    demand,
    dlm,
    dlm_choice,
    dlm_decomposition,
    dlm_diagnostics,
    dlm_lift,
    dlm_media,
    geo_dlm,
    nested,
    ridge_mmm,
    ridge_search,
)
from mixdyn.transforms import media, saturation

__all__ = ["FORMAT", "FORMAT_VERSION", "REGISTRY", "FormatError", "load", "save"]

FORMAT = "mixdyn"
"""The name a file's manifest gives its format."""
FORMAT_VERSION = 1
"""The version of the format this mixdyn writes, and the one version it reads."""
_MANIFEST = "manifest.json"
_ARRAYS = "arrays"

_CLASSES: tuple[type, ...] = (
    # the panel and the tests it is read with
    panel.Cadence,
    panel.Revenue,
    panel.Count,
    panel.Allotment,
    panel.LeverKind,
    panel.Baseline,
    panel.Lever,
    issues.Severity,
    issues.IssueCode,
    issues.Issue,
    feedback.SpendFeedback,
    recurring.RecurringEvents,
    panel.MediaPanel,
    lift.LiftEstimate,
    lift.LiftSeries,
    chc.lift.LiftTest,
    chc.lift.GeoArm,
    documents.Estimand,
    documents.Unfilled,
    documents.Source,
    documents.Prior,
    documents.Holdout,
    documents.ReturnRange,
    documents.PowerSummary,
    documents.Refresh,
    documents.Factsheet,
    documents.Spend,
    documents.Health,
    documents.Period,
    documents.Reconciliation,
    documents.NextTest,
    documents.Event,
    documents.Compute,
    documents.Decision,
    # transforms
    media.Geometric,
    media.WeibullCdf,
    media.Hill,
    saturation.TanhShape,
    saturation.HillShape,
    saturation.Transform,
    # the ridge engine
    ridge_mmm.Media,
    ridge_mmm.Control,
    demand.DemandProxy,
    ridge_mmm.LinearTrend,
    ridge_mmm.FourierSeason,
    ridge_mmm.RidgeSpec,
    ridge_mmm.Calibration,
    ridge_mmm.RidgeFit,
    ridge_search.Interval,
    ridge_search.GeometricSpace,
    ridge_search.WeibullSpace,
    ridge_search.MediaSpace,
    ridge_search.RidgeSpace,
    ridge_search.Folds,
    ridge_search.Optimum,
    ridge_search.RidgeSearch,
    sensitivity.Benchmark,
    sensitivity.BiasBound,
    sensitivity.RidgeSensitivity,
    # the Bayesian engines
    nuts.Chains,
    Loo,
    Psis,
    diagnostics.Diagnostics,
    diagnostics.HealthCard,
    diagnostics.PriorPredictive,
    sensitivity.MsmSensitivity,
    bayes_mmm.RoiPrior,
    bayes_mmm.HalfNormalRoi,
    bayes_mmm.BayesMedia,
    bayes_mmm.Normal,
    bayes_mmm.StudentT,
    bayes_mmm.LogNormal,
    bayes_mmm.HillCurve,
    bayes_mmm.TanhCurve,
    bayes_mmm.UniformStart,
    bayes_mmm.PathfinderStart,
    bayes_mmm.NaturalCoordinates,
    bayes_mmm.QRCoordinates,
    bayes_mmm.BayesSpec,
    bayes_mmm.BayesFit,
    bayes_mmm.BayesForecast,
    bayes_mmm.BayesHoldout,
    bayes_mmm.ReturnDrift,
    bayes_mmm.BayesRefresh,
    bayes_geo.BayesGeoSpec,
    bayes_geo.BayesGeoFit,
    nested.NestedSpec,
    nested.NestedFit,
    # the DLMs, and what causal-hybrid-control's filter hands them
    chc.dlm.Polynomial,
    chc.dlm.Seasonal,
    chc.dlm.Regression,
    chc.dlm.Prior,
    chc.dlm.DynamicLinearModel,
    chc.dlm.GeoDLM,
    chc.dlm.DLMFit,
    chc.dlm.GeoDLMFit,
    chc.dlm.SmoothedStates,
    chc.dlm.ConstrainedDraws,
    chc.dlm.GeoSpread,
    dlm.Trend,
    dlm.Season,
    dlm.Effect,
    dlm.DlmSpec,
    dlm.DlmFit,
    dlm.DlmForecast,
    dlm.DlmForecastDraws,
    dlm_lift.LiftReading,
    dlm_lift.Lifted,
    dlm_media.TransformGrid,
    dlm_choice.PerEffect,
    dlm_choice.Edge,
    dlm_choice.DiscountGrid,
    dlm_choice.Bounds,
    dlm_choice.ComponentBounds,
    dlm_choice.DlmChoice,
    dlm_decomposition.Components,
    dlm_decomposition.DlmDecomposition,
    dlm_diagnostics.Statistic,
    dlm_diagnostics.CheckCode,
    dlm_diagnostics.Finding,
    dlm_diagnostics.DlmDiagnostics,
    geo_dlm.GeoDlmSpec,
    geo_dlm.GeoDlmFit,
    geo_dlm.GeoEffects,
    # decisions
    chc.allocation.ReturnTarget,
    chc.allocation.MarginalReturnTarget,
    chc.allocation.ReturnOnSpendTarget,
    Returns,
    Costs,
    Undefined,
    geo_plan.Range,
    geo_plan.GeoLimits,
    geo_plan.GeoPlan,
    posterior.Readings,
    posterior.Worths,
    posterior.PosteriorPlan,
    posterior.CvarPlan,
    posterior.Breakdown,
    posterior.RobustPlan,
    Spread,
    PlanTable,
    scenarios.Scenario,
    scenarios.ScenarioComparison,
    scenarios.ResponseCurves,
    flighting.FlightPlan,
    # a geo test's reading
    reading.GeoTest,
    reading.GeoReading,
    # experiments: a test's design and its reading, and what a test is worth to a plan
    geo_design.Blocks,
    geo_design.GeoDesign,
    geo_design.Balance,
    geo_design.Power,
    geo_design.PowerAudit,
    geo_design.StratifiedDesign,
    sdid.SdidFit,
    sdid.SdidReading,
    switchback.SwitchbackDesign,
    switchback.SwitchbackReading,
    switchback.CarryoverOrder,
    candidates.After,
    candidates.Within,
    candidates.GroupNoise,
    candidates.PlaceboNoise,
    candidates.Candidate,
    experiment_value.Gross,
    experiment_value.Net,
    experiment_value.Reweight,
    experiment_value.Gaussian,
    experiment_value.ExperimentValue,
    experiment_value.ExperimentValues,
    experiment_value.ExperimentChoice,
)


def _name(cls: type) -> str:
    return f"{cls.__module__}:{cls.__qualname__}"


REGISTRY: Mapping[str, type] = MappingProxyType({_name(cls): cls for cls in _CLASSES})
"""Every class a file may hold, by `module:qualname`."""


class FormatError(ValueError):
    """A file that is not one this mixdyn reads: not an archive of the format, of another format
    version, or naming a class or a value the format does not hold."""


def save(obj: object, path: str | Path) -> None:
    """Write `obj`, an instance of a class in `REGISTRY`, to `path` as the format's archive.

    Raises:
        TypeError: on a value of a class the format does not hold, anywhere in `obj`, named by
            where it sits; and on an array of Python objects.
    """
    if _name(type(obj)) not in REGISTRY:
        msg = f"{_name(type(obj))} is not a class the format saves; see REGISTRY"
        raise TypeError(msg)
    arrays: list[np.ndarray] = []
    root = _encode(obj, arrays, "obj")
    manifest = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "mixdyn": metadata.version("mixdyn"),
        "causal-hybrid-control": metadata.version("causal-hybrid-control"),
        "root": root,
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(_MANIFEST, json.dumps(manifest, allow_nan=False, sort_keys=True))
        for i, array in enumerate(arrays):
            buffer = io.BytesIO()
            np.save(buffer, array, allow_pickle=False)
            archive.writestr(f"{_ARRAYS}/{i}.npy", buffer.getvalue())


def load(path: str | Path) -> Any:
    """The object `save` wrote to `path`.

    Raises:
        FormatError: on a file that is not the format's archive or whose members do not read, a
            format version other than `FORMAT_VERSION`, a class or a node the format does not
            hold, a value its class refuses, and a JAX array in double precision where JAX runs
            in single.
    """
    try:
        archive = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, NotImplementedError) as error:
        msg = f"{path} is not a {FORMAT} file: it is not a zip archive that Python reads"
        raise FormatError(msg) from error
    with archive:
        try:
            manifest = json.loads(_read(archive, _MANIFEST, str(path)))
        except (KeyError, ValueError) as error:
            msg = f"{path} is not a {FORMAT} file: it has no readable {_MANIFEST}"
            raise FormatError(msg) from error
        if not isinstance(manifest, dict) or manifest.get("format") != FORMAT:
            msg = f"{path} is not a {FORMAT} file: its manifest names no format {FORMAT!r}"
            raise FormatError(msg)
        version = manifest.get("format_version")
        if isinstance(version, bool) or not isinstance(version, int):
            msg = (
                f"{path} has no valid format version, {version!r}: the file is damaged or was "
                f"not written by mixdyn's `save`"
            )
            raise FormatError(msg)
        if version != FORMAT_VERSION:
            later = "a later" if version > FORMAT_VERSION else "an earlier"
            msg = (
                f"{path} is format version {version}, written by mixdyn "
                f"{manifest.get('mixdyn', '(unknown)')}; this mixdyn, "
                f"{metadata.version('mixdyn')}, reads format version {FORMAT_VERSION} only. "
                f"Load it with {later} mixdyn"
            )
            raise FormatError(msg)
        if "root" not in manifest:
            msg = f"{path} is not a {FORMAT} file: its manifest holds no object"
            raise FormatError(msg)
        return _decode(manifest["root"], archive, "obj")


def _encode(value: Any, arrays: list[np.ndarray], where: str) -> Any:
    if isinstance(value, enum.Enum):
        return {"$enum": _registered(type(value), where), "member": value.name}
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        return value if math.isfinite(value) else {"$float": repr(value)}
    if isinstance(value, jax.Array):
        return {"$array": _stored(np.asarray(value), arrays, where), "jax": True}
    if isinstance(value, np.ndarray):
        node: dict[str, Any] = {"$array": _stored(value, arrays, where)}
        if not value.flags.writeable:
            node["readonly"] = True
        return node
    if isinstance(value, np.generic):
        return {"$scalar": _stored(np.asarray(value), arrays, where)}
    if type(value) is dt.date:
        return {"$date": value.isoformat()}
    if type(value) is tuple:
        return {"$tuple": [_encode(v, arrays, f"{where}[{i}]") for i, v in enumerate(value)]}
    if type(value) is list:
        return {"$list": [_encode(v, arrays, f"{where}[{i}]") for i, v in enumerate(value)]}
    if type(value) is frozenset:
        items = [_encode(v, arrays, f"{where} member") for v in value]
        return {"$frozenset": sorted(items, key=lambda node: json.dumps(node, sort_keys=True))}
    if type(value) in (dict, MappingProxyType):
        items = [
            [_encode(k, arrays, f"{where} key"), _encode(v, arrays, f"{where}[{k!r}]")]
            for k, v in value.items()
        ]
        return {"$dict": items, "proxy": type(value) is MappingProxyType}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        fields = {
            f.name: _encode(getattr(value, f.name), arrays, f"{where}.{f.name}")
            for f in dataclasses.fields(value)
            if f.init
        }
        return {"$type": _registered(type(value), where), "fields": fields}
    msg = f"{where} is a {_name(type(value))}, which the format does not save"
    raise TypeError(msg)


def _registered(cls: type, where: str) -> str:
    name = _name(cls)
    if name not in REGISTRY:
        msg = f"{where} is a {name}, which the format does not save; see REGISTRY"
        raise TypeError(msg)
    return name


def _stored(array: np.ndarray, arrays: list[np.ndarray], where: str) -> int:
    if array.dtype.hasobject:
        msg = f"{where} is an array of Python objects, which the format does not save"
        raise TypeError(msg)
    arrays.append(array)
    return len(arrays) - 1


def _decode(node: Any, archive: zipfile.ZipFile, where: str) -> Any:
    if node is None or isinstance(node, bool | int | float | str):
        return node
    if not isinstance(node, dict) or not node or not next(iter(node)).startswith("$"):
        raise FormatError(f"{where} is not a node of the format: {_short(node)}")
    if "$float" in node:
        return float(node["$float"])
    if "$array" in node:
        array = _array(node["$array"], archive, where)
        if node.get("jax"):
            if array.dtype.itemsize == 8 and not jax.config.read("jax_enable_x64"):
                msg = (
                    f"{where} is a JAX array of {array.dtype}, which JAX in single precision would "
                    "round: enable `jax_enable_x64` before loading, as the engine runs"
                )
                raise FormatError(msg)
            return jnp.asarray(array)
        if node.get("readonly"):
            array.flags.writeable = False
        return array
    if "$scalar" in node:
        return _array(node["$scalar"], archive, where)[()]
    if "$date" in node:
        return dt.date.fromisoformat(node["$date"])
    if "$tuple" in node:
        return tuple(_decode(v, archive, f"{where}[{i}]") for i, v in enumerate(node["$tuple"]))
    if "$list" in node:
        return [_decode(v, archive, f"{where}[{i}]") for i, v in enumerate(node["$list"])]
    if "$frozenset" in node:
        return frozenset(_decode(v, archive, f"{where} member") for v in node["$frozenset"])
    if "$dict" in node:
        items = {
            _decode(k, archive, f"{where} key"): _decode(v, archive, f"{where}[{k!r}]")
            for k, v in node["$dict"]
        }
        return MappingProxyType(items) if node.get("proxy") else items
    if "$enum" in node:
        cls = _class(node["$enum"], where)
        if not issubclass(cls, enum.Enum) or node.get("member") not in cls.__members__:
            raise FormatError(f"{where}: {node['$enum']} has no member {node.get('member')!r}")
        return cls[node["member"]]
    if "$type" in node:
        cls = _class(node["$type"], where)
        if not dataclasses.is_dataclass(cls):
            raise FormatError(f"{where}: {node['$type']} is not a dataclass")
        fields = {
            name: _decode(v, archive, f"{where}.{name}")
            for name, v in node.get("fields", {}).items()
        }
        try:
            return cls(**fields)
        # The class's checks are the format's: they refuse a value with `ValueError`, and a value
        # of a type other than its field's, which no check reads first, with whatever error their
        # code meets first.
        except Exception as error:
            msg = f"{where}: {node['$type']} cannot be rebuilt: {type(error).__name__}: {error}"
            raise FormatError(msg) from error
    raise FormatError(f"{where} is not a node of the format: {_short(node)}")


def _class(name: Any, where: str) -> type:
    cls = REGISTRY.get(name) if isinstance(name, str) else None
    if cls is None:
        msg = f"{where} names a class the format does not hold, {name!r}; see REGISTRY"
        raise FormatError(msg)
    return cls


def _read(archive: zipfile.ZipFile, name: str, where: str) -> bytes:
    """The bytes of the archive's member `name`.

    Raises:
        KeyError: where the archive holds no member `name`.
        FormatError: where the member does not read: a checksum or a header that is damaged,
            data that does not decompress, or a compression or an encryption that the zip module
            does not read.
    """
    try:
        return archive.read(name)
    except KeyError:
        raise
    # A damaged member raises the error of whichever layer finds the damage: zipfile's own
    # `BadZipFile`, `ValueError`, `RuntimeError` or `NotImplementedError`, or its decompressor's,
    # `zlib.error`, `OSError` from bz2, `LZMAError`, or `ZstdError` from Python 3.14. They share
    # no base class short of `Exception`.
    except Exception as error:
        raise FormatError(f"{where}: {name} in the archive does not read: {error}") from error


def _array(index: Any, archive: zipfile.ZipFile, where: str) -> np.ndarray:
    try:
        data = _read(archive, f"{_ARRAYS}/{index}.npy", where)
    except KeyError as error:
        raise FormatError(f"{where}: the archive has no array {index!r}") from error
    try:
        return np.load(io.BytesIO(data), allow_pickle=False)
    except ValueError as error:
        raise FormatError(f"{where}: array {index!r} is damaged: {error}") from error


def _short(node: Any) -> str:
    text = repr(node)
    return text if len(text) <= 80 else f"{text[:77]}..."
