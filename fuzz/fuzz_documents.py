"""Fuzz `mixdyn.api`'s readers of saved bytes: `loads`, of a JSON document, and `serialize.load`.

    uv run --with-requirements fuzz/requirements.txt \
        python fuzz/fuzz_documents.py -max_total_time=60

Each input decodes into a text for `loads` or a file for `load`. Most start as one the engine
writes, of a value made here as the tests make theirs:

* a document `dumps` writes of `ISSUES`, `LIFT_TESTS`, `FACTSHEET` or `DECISION`, or of a panel or
  an array under a schema of the target's own, as the tests read them;
* a file `save` writes of a value a caller saves: an issue, a lift test and its estimand, a plan
  over geos, a panel, returns that hold NumPy scalars, non-finite floats or a JAX array, and
  others, between them every node of the format.

Then up to four of its JSON values are replaced, each by any JSON value (an integer of any size, a
float that overflows, `NaN`, `Infinity`, a date that is none, text, a list, an object nested
thousands deep, or a node of the format copied from elsewhere in the original), dropped, given a
sibling, or renamed; a file's array is replaced by `.npy` bytes of another dtype, shape or header,
by an `.npz` or by the input's bytes; its members are stored or compressed; and the text or the
file's bytes may be cut or overwritten. The last kind of input is the input's bytes alone, as the
text or the file.

`loads` refuses the text with `SchemaError`, and `load` the file with `FormatError`, or each
returns a value that keeps its module's promise:

* `dumps` writes the value again, and `loads` reads that back the same: each field, item, key and
  array, NaN where NaN;
* `save` writes the value again, and `load` reads that back the same to the bit.

Anything else, a broken promise or another exception, is a crash with the input that caused it.
"""

from __future__ import annotations

import atexit
import copy
import dataclasses
import datetime as dt
import enum
import io
import json
import math
import shutil
import struct
import sys
import tempfile
import zipfile
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import atheris
import jax
import jax.numpy as jnp
import numpy as np
from atheris.import_hook import HookManager

# Branch coverage of the whole package, without atheris's comparison tracing: in atheris 3.1.0
# that tracing keeps each numpy array a traced comparison returns (41 MB over 3 000 inputs of
# the planners), and `instrument_imports` always turns it on.
with HookManager({"mixdyn"}, set(), enable_loader_override=True, trace_dataflow=False):
    from mixdyn import (
        NATIONAL,
        Baseline,
        Cadence,
        Channel,
        Count,
        Geo,
        Issue,
        IssueCode,
        Lever,
        LeverKind,
        MediaPanel,
        Revenue,
        Severity,
    )
    from mixdyn.api import (
        DECISION,
        FACTSHEET,
        ISSUES,
        LIFT_TESTS,
        Compute,
        Decision,
        Estimand,
        Event,
        Factsheet,
        Health,
        Holdout,
        NextTest,
        Period,
        PowerSummary,
        Prior,
        Reconciliation,
        Refresh,
        ReturnRange,
        Schema,
        SchemaError,
        Source,
        Spend,
        Unfilled,
        dumps,
        loads,
    )
    from mixdyn.api.serialize import REGISTRY, FormatError, load, save
    from mixdyn.calibration import LiftEstimate
    from mixdyn.data import RecurringEvents
    from mixdyn.decisions import Costs, GeoPlan, PosteriorPlan, Returns, Undefined
    from mixdyn.experiments import After, Candidate, PlaceboNoise
    from mixdyn.models import TransformGrid
    from mixdyn.models.bayes_mmm import BayesMedia, HalfNormalRoi
    from mixdyn.transforms import HillShape, TanhShape

jax.config.update("jax_enable_x64", True)  # as the engine runs, and as the tests load

PANEL = Schema("fuzz.panel", 1, MediaPanel)
ARRAY = Schema("fuzz.array", 1, np.ndarray)


# ----------------------------------------------------------------------------- what is written


def _panel() -> MediaPanel:
    """Two geos over four weeks, a channel, a control and a population: the tests' panel, small."""
    return MediaPanel(
        dates=np.datetime64("2024-01-01") + np.timedelta64(7, "D") * np.arange(4),
        cadence=Cadence.WEEK,
        geos=(Geo("north"), Geo("south")),
        channels=(Channel("tv"),),
        controls=("price",),
        kpi=np.array([[10.0, 11.0, np.nan, 12.5], [7.0, 6.5, 7.25, 8.0]]),
        kpi_unit=Count("orders"),
        spend=np.array([[[1.0], [2.0], [0.0], [0.5]], [[0.0], [1.5], [3.0], [0.25]]]),
        currency="EUR",
        control_values=np.array([[[1.0], [1.1], [np.nan], [0.9]], [[2.0], [2.0], [2.1], [1.9]]]),
        filled=np.array([[False, False, True, False], [False, False, False, False]]),
        population=np.array([2.0e5, 3.0e5]),
    )


ESTIMANDS = (
    Estimand("tv", (), 0.2, "added", "marginal change", 4, 8, 1.5, 1.0, None, None, None, "a test"),
    Estimand(
        "search",
        ("north",),
        1.0,
        "withheld",
        "lift against zero spend",
        2,
        4,
        0.0,
        100.0,
        dt.date(2024, 1, 13),
        dt.date(2024, 1, 20),
        dt.date(2024, 2, 3),
        "search, in north: spend withheld",
    ),
)
ISSUED = (
    Issue(IssueCode.DUPLICATE, Severity.ERROR, "twice", geo="north", dates=(dt.date(2024, 1, 6),)),
    Issue(IssueCode.CONSTANT, Severity.WARNING, "flat", None, "retail_eur", (), "retail"),
)


def _factsheet() -> Factsheet:
    return Factsheet(
        model="a Bayesian media-mix model",
        kpi="orders",
        cadence="week",
        first=dt.date(2024, 1, 1),
        last=dt.date(2024, 12, 30),
        geos=("north", "south"),
        channels=("tv", "search"),
        sources=(Source("sales", "kpi", "a warehouse", ("summed by week",)),),
        priors=(Prior("tv's return", "HalfNormal(2)", "a lift test"),),
        tests=ESTIMANDS,
        holdout=Holdout(dt.date(2024, 11, 4), dt.date(2024, 12, 30), 9, "MAPE", 0.12),
        returns=(ReturnRange("tv", "window", 1.4, 0.8, 2.1, 0.9),),
        power=PowerSummary(12, 1.5, 4.0),
        refresh=Refresh(4, "week", dt.date(2024, 12, 30)),
        limitations=("no price data",),
    )


def _decision() -> Decision:
    return Decision(
        title="next quarter",
        geo="national",
        currency="EUR",
        periods=13,
        budget=1300.0,
        gain=420.5,
        current_gain=400.0,
        spend=(Spend("tv", 60.0, 70.0), Spend("search", 40.0, 30.0)),
        health=Health(
            max_rhat=1.01,
            ppp=0.4,
            ppp_se=0.01,
            negative_baseline=0.0,
            shift={"tv": 0.1},
            width={"tv": 0.5},
            consistency={"tv": math.nan},
            thresholds={"max_rhat": 1.05},
            flags={"max_rhat": True},
        ),
        reconciliation=Reconciliation(
            ("base", "tv"), (Period(dt.date(2024, 1, 1), 10.0, (8.0, 1.5), 9.5, 0.5),)
        ),
        next_test=NextTest("test", ("tv dark",), ("tv",), 3.0, 0.5, 1.0, 2.0, 2.5, 4.0, 0.2, 400),
        events=(Event(dt.date(2024, 1, 2), "planned"),),
        compute=Compute("cpu", 12.5),
        unidentified=("search",),
    )


DOCUMENTS: tuple[tuple[Schema, Any], ...] = (
    (ISSUES, ISSUED),
    (LIFT_TESTS, ESTIMANDS),
    (FACTSHEET, _factsheet()),
    (DECISION, _decision()),
    (
        DECISION,
        dataclasses.replace(
            _decision(),
            health=Unfilled("no fit"),
            compute=Unfilled("not recorded"),
            unidentified=(),
        ),
    ),
    (PANEL, _panel()),
    (ARRAY, np.array([["2024-01-01", "NaT"]], dtype="datetime64[D]")),
    (ARRAY, np.arange(6, dtype=np.int64).reshape(2, 3)),
)
"""Each schema and a value of it, as `dumps` writes it."""

HELD: tuple[Any, ...] = (
    ISSUED[0],
    ESTIMANDS[1],
    LiftEstimate(
        np.zeros(4),
        np.ones(4),
        2,
        -1.0,
        0.5,
        0.25,
        frozenset({Geo("b")}),
        np.datetime64("2024-01-27"),
    ),
    Costs(120.0, 40.0, 3.0, Undefined("one more currency unit earns no incremental KPI")),
    Returns(np.str_("tv"), np.float64(1.5), np.int64(2), math.inf, -0.0, math.nan),  # ty: ignore[invalid-argument-type]
    Returns(jnp.arange(3.0), 0.0, 0.0, 0.0, 0.0, 0.0),  # ty: ignore[invalid-argument-type]
    After({Channel("tv"): 50.0}),
    PlaceboNoise(np.linspace(-1.0, 1.0, 12), pre=4),
    Candidate(Channel("tv"), np.array([0.0, 0.0]), np.array([1.0, 1.0]), 2, 0.5, 0.2, "dark"),
    Lever(LeverKind.PRICE, Baseline.MAX),
    RecurringEvents(("weeks 47-48",), ((47, 48),), np.linspace(0.0, 1.0, 52), 3.5),
    TransformGrid((0.0, 0.5), (1.0, 2.0), (TanhShape(), HillShape(1.5))),
    BayesMedia("tv", HalfNormalRoi(2.0, counts="carried")),
    PosteriorPlan(
        (Channel("tv"), Channel("retail")),
        4,
        np.array([60.0, 40.0]),
        np.array([50.0, 80.0]),
        400.0,
        12.5,
        10.0,
        64,
        (Channel("retail"),),
    ),
    GeoPlan(
        geos=(NATIONAL,),
        channels=(Channel("tv"), Channel("retail")),
        periods=4,
        spend=np.array([[60.0, 40.0]]),
        current=np.array([[50.0, 80.0]]),
        budget=400.0,
        gain=12.5,
        current_gain=10.0,
        shortfall=0.0,
        price=0.5,
        geo_prices=MappingProxyType({NATIONAL: 0.0}),
        channel_prices=MappingProxyType({Channel("tv"): 0.0, Channel("retail"): 0.0}),
        unidentified=((NATIONAL, Channel("retail")),),
    ),
    dataclasses.replace(_panel(), kpi_unit=Revenue()),
    _decision(),
)
"""Values a caller saves, between them every node of the format."""

_WORK = Path(tempfile.mkdtemp(prefix="fuzz-documents-"))
atexit.register(shutil.rmtree, _WORK, ignore_errors=True)
FILE, AGAIN = _WORK / "read.mixdyn", _WORK / "again.mixdyn"


def _archive(value: Any) -> tuple[Any, dict[str, bytes]]:
    """The manifest `save` writes for `value`, as a tree, and its other members."""
    save(value, FILE)
    with zipfile.ZipFile(FILE) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    return json.loads(members.pop("manifest.json")), members


ARCHIVES = tuple(_archive(value) for value in HELD)
TEXTS = tuple(json.loads(dumps(value, schema)) for schema, value in DOCUMENTS)


def _npy(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def _npz() -> bytes:
    buffer = io.BytesIO()
    np.savez(buffer, a=np.arange(3.0))
    return buffer.getvalue()


ARRAYS = (
    _npy(np.array([1.5, -2.5], dtype=np.float32)),
    _npy(np.array([-3, 4], dtype=np.int8)),
    _npy(np.array([2**64 - 1], dtype=np.uint64)),
    _npy(np.array([1.0 + 2.0j])),
    _npy(np.array([0.5], dtype=np.float16)),
    _npy(np.array(["abc", ""])),
    _npy(np.array([b"ab"], dtype="S2")),
    _npy(np.zeros(2, dtype="V4")),
    _npy(np.zeros(2, dtype=[("a", "<f8"), ("b", "<i4")])),
    _npy(np.array(["2024-01-01", "NaT"], dtype="datetime64[D]")),
    _npy(np.array([3, -1], dtype="timedelta64[s]")),
    _npy(np.array([True, False])),
    _npy(np.array([1.0, 2.0], dtype=">f8")),
    _npy(np.array(3.0)),
    _npy(np.array([], dtype=np.float64)),
    _npy(np.zeros((2, 0, 3))),
    _npy(np.asfortranarray(np.arange(6.0).reshape(2, 3))),
    _npy(np.array([math.nan, math.inf, -0.0, 5e-324])),
    _npy(np.arange(64.0)),
    _npz(),
    b"\x80\x04\x95\x05\x00\x00\x00\x00\x00\x00\x00K\x01.",  # a pickle of 1
)
"""`.npy` bytes of every kind, an `.npz` archive, and a pickle, to stand for an array."""

DESCRS = (
    "'<f8'",
    "'|O'",
    # wider than any payload here, though no wider: NumPy casts even an empty array of it to
    # float through a buffer of 128 items, 51 MB
    "'<U100000'",
    "'|V0'",
    "'<f8', 'x': 1",
    "[('a', '<f8')]",
    "'<M8[D]'",
    "'>c16'",
    "'bogus'",
    "('<f8', (2,))",
)
SHAPES = ((), (0,), (3,), (2**31,), (2**40,), (2**62, 4), (-1,), (2, -2), (10**30,))
INTS = (0, -1, 1, 2, 2**31, 2**53 + 1, 2**63 - 1, 2**63, -(2**63) - 1, 2**64, 10**308, 10**400)
FLOATS = (
    -0.0,
    5e-324,
    1.7976931348623157e308,
    1e-310,
    math.nan,
    math.inf,
    -math.inf,
)  # json.dumps writes the last three as NaN, Infinity and -Infinity, which JSON is not
RAWS = (
    "1e400",
    "-1e400",
    "1E999999999",
    "-0",
    "0.0e-999999",
    "1." + "0" * 30 + "1",
    "1" * 4400,
    "-" + "9" * 4300,
    "NaN",
    "-Infinity",
    "[" + ", ".join(["1" + "0" * 400] * 11) + "]",
)
"""Numbers written as JSON text: past a float's range, past Python's conversion limit, or not
JSON; and a shape whose product is past that limit."""
STRINGS = (
    "",
    " ",
    "2024-01-06",
    "2024-02-30",
    "0001-01-01",
    "9999-12-31",
    "+10000-01-01",
    "-0001-01-01",
    "20240106",
    "2024-W01-1",
    "2024-01-06T00:00",
    "NaT",
    "nat",
    "today",
    "now",
    "nan",
    "inf",
    "-inf",
    "Infinity",
    "1e400",
    "0x1p-1074",
    "\ud800",
    "float64",
    "int64",
    "bool",
    "datetime64[D]",
    "Unfilled",
    "Holdout",
    "Count",
    "Revenue",
    "duplicate",
    "error",
    "withheld",
    "plan now",
    "arrays/0.npy",
    "../manifest.json",
    "mixdyn",
)
KEYS = (
    "$float",
    "$array",
    "$scalar",
    "$date",
    "$tuple",
    "$list",
    "$frozenset",
    "$dict",
    "$enum",
    "$type",
    "jax",
    "readonly",
    "proxy",
    "member",
    "fields",
    "kind",
    "dtype",
    "shape",
    "data",
    "schema",
    "version",
    "root",
    "format",
    "format_version",
)
NAMES = tuple(REGISTRY)
COMPRESSIONS = (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED, zipfile.ZIP_BZIP2, zipfile.ZIP_LZMA)


# ------------------------------------------------------------------------- editing a JSON tree


class _Raw:
    """Text to stand in the written JSON as it is, which `json.dumps` cannot write."""

    def __init__(self, text: str) -> None:
        self.text = text


def _raw(provider: atheris.FuzzedDataProvider) -> _Raw:
    kind = provider.ConsumeIntInRange(0, 2)
    if kind == 0:
        return _Raw(RAWS[provider.ConsumeIntInRange(0, len(RAWS) - 1)])
    depth = provider.ConsumeIntInRange(1, 200_000)
    if kind == 1:
        return _Raw("[" * depth + "]" * depth)
    return _Raw('{"$tuple": [' * depth + "1" + "]}" * depth)


def _value(provider: atheris.FuzzedDataProvider, seed: Any, depth: int = 0) -> Any:
    """Any JSON value, a node of the format, or a copy of a value from `seed`."""
    kind = provider.ConsumeIntInRange(0, 13)
    if kind == 0:
        return None
    if kind == 1:
        return provider.ConsumeBool()
    if kind == 2:
        return provider.ConsumeIntInRange(-3, 12)
    if kind == 3:
        return INTS[provider.ConsumeIntInRange(0, len(INTS) - 1)]
    if kind == 4:
        return provider.ConsumeRegularFloat()
    if kind == 5:
        return FLOATS[provider.ConsumeIntInRange(0, len(FLOATS) - 1)]
    if kind == 6:
        return provider.ConsumeUnicode(provider.ConsumeIntInRange(0, 8))
    if kind == 7:
        return STRINGS[provider.ConsumeIntInRange(0, len(STRINGS) - 1)]
    if kind == 8:
        return NAMES[provider.ConsumeIntInRange(0, len(NAMES) - 1)]
    if kind == 9 and depth < 3:
        return [_value(provider, seed, depth + 1) for _ in range(provider.ConsumeIntInRange(0, 3))]
    if kind == 10 and depth < 3:
        return {
            KEYS[provider.ConsumeIntInRange(0, len(KEYS) - 1)]: _value(provider, seed, depth + 1)
            for _ in range(provider.ConsumeIntInRange(0, 3))
        }
    if kind == 11 and depth == 0:
        return _raw(provider)
    slots = _slots([seed])
    holder, key = slots[provider.ConsumeIntInRange(0, len(slots) - 1)]
    return copy.deepcopy(holder[key])


def _slots(root: list[Any]) -> list[tuple[Any, Any]]:
    """Every place in the tree `root` holds that holds a value, as its holder and its key there,
    `root` itself first."""
    found: list[tuple[Any, Any]] = []
    stack: list[tuple[Any, Any]] = [(root, 0)]
    while stack:
        holder, key = stack.pop()
        found.append((holder, key))
        value = holder[key]
        if isinstance(value, dict):
            stack.extend((value, k) for k in value)
        elif isinstance(value, list):
            stack.extend((value, i) for i in range(len(value)))
    return found


def _edit(provider: atheris.FuzzedDataProvider, seed: Any) -> Any:
    """A copy of `seed` with up to four of its values replaced, dropped, given a sibling, or renamed
    under one of `KEYS`.

    A new value copies its parts from `seed`, never from the copy being edited, and nests text
    thousands deep only at its top: an input of a kilobyte writes a few times `seed` and four such
    texts at most. Copies of the copy being edited would grow as a power of the input's length.
    """
    root = [copy.deepcopy(seed)]
    for _ in range(provider.ConsumeIntInRange(0, 4)):
        slots = _slots(root)
        holder, key = slots[provider.ConsumeIntInRange(0, len(slots) - 1)]
        kind = provider.ConsumeIntInRange(0, 5)
        if kind <= 2 or holder is root:
            holder[key] = _value(provider, seed)
        elif kind == 3:
            del holder[key]
        elif kind == 4 and isinstance(holder, dict):
            holder[KEYS[provider.ConsumeIntInRange(0, len(KEYS) - 1)]] = _value(provider, seed)
        elif kind == 4:
            holder.insert(key, _value(provider, seed))
        elif isinstance(holder, dict):
            holder[KEYS[provider.ConsumeIntInRange(0, len(KEYS) - 1)]] = holder.pop(key)
    return root[0]


def _written(tree: Any) -> str:
    """`tree` as JSON text, each `_Raw` written as it is."""
    raws: list[str] = []

    def mark(raw: Any) -> str:
        if not isinstance(raw, _Raw):
            raise TypeError(type(raw).__name__)
        raws.append(raw.text)
        return f"\x00{len(raws) - 1}\x00"

    text = json.dumps(tree, default=mark)
    for i, raw in enumerate(raws):
        text = text.replace(json.dumps(f"\x00{i}\x00"), raw, 1)
    return text


def _damaged(provider: atheris.FuzzedDataProvider, data: bytes) -> bytes:
    """`data` cut, or a few of its bytes overwritten, or as it is."""
    kind = provider.ConsumeIntInRange(0, 7)
    if kind == 0 and data:
        return data[: provider.ConsumeIntInRange(0, len(data) - 1)]
    if kind == 1 and data:
        at = provider.ConsumeIntInRange(0, len(data) - 1)
        patch = provider.ConsumeBytes(provider.ConsumeIntInRange(1, 4))
        return data[:at] + patch + data[at + len(patch) :]
    return data


# ---------------------------------------------------------------------------------- the checks


def _differences(a: Any, b: Any, where: str = "value") -> list[str]:
    """Where `a` and `b` differ: in type, a field, an item, a key, an array's dtype, shape or
    writeability, or any bit of a value."""
    if type(a) is not type(b) and not (isinstance(a, jax.Array) and isinstance(b, jax.Array)):
        return [f"{where}: {type(a).__name__} against {type(b).__name__}"]
    if isinstance(a, np.ndarray) and a.flags.writeable != b.flags.writeable:
        return [f"{where}: writeable {a.flags.writeable} against {b.flags.writeable}"]
    if isinstance(a, jax.Array | np.ndarray | np.generic):
        a, b = np.asarray(a), np.asarray(b)
        if a.dtype != b.dtype or a.shape != b.shape:
            return [f"{where}: {a.dtype}{a.shape} against {b.dtype}{b.shape}"]
        return [] if a.tobytes() == b.tobytes() else [f"{where}: {a!r} against {b!r}"]
    if isinstance(a, float):
        return [] if a.hex() == b.hex() else [f"{where}: {a!r} against {b!r}"]
    if isinstance(a, tuple | list):
        if len(a) != len(b):
            return [f"{where}: {len(a)} items against {len(b)}"]
        return [
            d
            for i, (x, y) in enumerate(zip(a, b, strict=True))
            for d in _differences(x, y, f"{where}[{i}]")
        ]
    if isinstance(a, frozenset):
        unmatched = [x for x in a if not any(not _differences(x, y) for y in b)]
        return [f"{where}: {x!r} has no match in {b!r}" for x in unmatched] + (
            [f"{where}: {len(a)} members against {len(b)}"] if len(a) != len(b) else []
        )
    if isinstance(a, Mapping):
        if keys := _differences(list(a), list(b), f"{where} keys"):
            return keys
        return [
            d
            for (k, x), y in zip(a.items(), b.values(), strict=True)
            for d in _differences(x, y, f"{where}[{k!r}]")
        ]
    if dataclasses.is_dataclass(a):
        return [
            d
            for f in dataclasses.fields(a)
            for d in _differences(getattr(a, f.name), getattr(b, f.name), f"{where}.{f.name}")
        ]
    if isinstance(a, enum.Enum | str | int | bool | dt.date) or a is None:
        return [] if a == b else [f"{where}: {a!r} against {b!r}"]
    return [f"{where}: no comparison for {type(a).__name__}"]


def _document(provider: atheris.FuzzedDataProvider) -> None:
    index = provider.ConsumeIntInRange(0, len(DOCUMENTS) - 1)
    schema = DOCUMENTS[index][0]
    kind = provider.ConsumeIntInRange(0, 7)
    if kind == 0:
        text = provider.ConsumeUnicode(provider.remaining_bytes())
    else:  # the schema's own document, or now and then another's
        tree = TEXTS[index if kind > 1 else provider.ConsumeIntInRange(0, len(TEXTS) - 1)]
        text = _written(_edit(provider, tree))
        if provider.ConsumeIntInRange(0, 3) == 0:
            text = _damaged(provider, text.encode()).decode(errors="replace")
    try:
        value = loads(text, schema)
    except SchemaError:
        return
    again = dumps(value, schema)
    back = loads(again, schema)
    if differences := _differences(value, back):
        raise AssertionError(f"{schema.name}: {again} reads back otherwise: {differences}")


def _npy_header(provider: atheris.FuzzedDataProvider) -> bytes:
    """A `.npy` file of version 1.0 whose header holds any dtype, order and shape."""
    descr = DESCRS[provider.ConsumeIntInRange(0, len(DESCRS) - 1)]
    shape = SHAPES[provider.ConsumeIntInRange(0, len(SHAPES) - 1)]
    header = f"{{'descr': {descr}, 'fortran_order': {provider.ConsumeBool()}, 'shape': {shape}, }}"
    header += " " * (63 - (10 + len(header)) % 64) + "\n"
    payload = provider.ConsumeBytes(provider.ConsumeIntInRange(0, 64))
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header.encode() + payload


def _file(provider: atheris.FuzzedDataProvider) -> None:
    kind = provider.ConsumeIntInRange(0, 7)
    if kind == 0:
        FILE.write_bytes(provider.ConsumeBytes(provider.remaining_bytes()))
    else:
        manifest, members = ARCHIVES[provider.ConsumeIntInRange(0, len(ARCHIVES) - 1)]
        manifest = _edit(provider, manifest)
        members = dict(members)
        for _ in range(provider.ConsumeIntInRange(0, 2)):
            name = f"arrays/{provider.ConsumeIntInRange(0, len(members))}.npy"
            change = provider.ConsumeIntInRange(0, 4)
            if change <= 1:
                members[name] = ARRAYS[provider.ConsumeIntInRange(0, len(ARRAYS) - 1)]
            elif change == 2:
                members[name] = _npy_header(provider)
            elif change == 3:
                members[name] = provider.ConsumeBytes(provider.ConsumeIntInRange(0, 128))
            else:
                members.pop(name, None)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            for name, data in {"manifest.json": _written(manifest).encode(), **members}.items():
                compression = COMPRESSIONS[provider.ConsumeIntInRange(0, len(COMPRESSIONS) - 1)]
                archive.writestr(name, data, compress_type=compression)
        FILE.write_bytes(_damaged(provider, buffer.getvalue()))
    try:
        value = load(FILE)
    except FormatError:
        return
    if type(value) not in REGISTRY.values():
        return  # `save` writes an instance of a class the format holds, and nothing else
    save(value, AGAIN)
    back = load(AGAIN)
    if differences := _differences(value, back):
        raise AssertionError(f"a file reads back otherwise: {differences}")


def check(data: bytes) -> None:
    provider = atheris.FuzzedDataProvider(data)
    if provider.ConsumeBool():
        _document(provider)
    else:
        _file(provider)


def main() -> None:
    atheris.Setup(sys.argv, check)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
