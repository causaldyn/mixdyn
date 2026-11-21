"""The kernels a digest of exact float64 bits was computed by, named as `Dispatch`.

A digest of a computation's exact bits is a fact of the code that ran it, and four parts of that
code are picked on the machine at run time: numpy picks its float64 `exp`, `log`, `tanh`, `expm1`
and `log1p` loops by the CPU's vector extensions; each OpenBLAS (numpy and scipy carry one each)
picks its kernels by the core type it detects; XLA compiles for the CPU name and the features LLVM
reads off the host; and the C library's `libm` answers every scalar call to it, Python's `math`
among them. The CPU's model name picks none of them: two models can share every kernel, and one
model can expose other extensions under another hypervisor.

`python tests/dispatch.py` prints this machine's key."""

from __future__ import annotations

import ctypes
import functools
import platform
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import jax
import scipy.linalg  # noqa: F401  # loads scipy's OpenBLAS beside numpy's
from jax._src.lib import xla_client
from numpy.lib.introspect import opt_func_info

KERNELS = ("exp", "log", "tanh", "expm1", "log1p")
"""The float64 loops numpy dispatches at run time that the engine calls, the last two in
`loo.psis`."""

_CORENAME = (
    "scipy_openblas_get_corename64_",
    "scipy_openblas_get_corename",
    "openblas_get_corename64_",
    "openblas_get_corename",
)
_SIMD = ("sse", "ssse", "avx", "fma", "f16c", "amx", "xop", "evex512")
"""Prefixes of the LLVM features that vector code is made of; the others (`aes`, `sha`, `rdrnd`,
...) no float64 kernel uses, and hypervisors mask them differently on the same CPU."""


@dataclass(frozen=True, slots=True)
class Dispatch:
    """What picks the kernels: numpy's per loop, each OpenBLAS's core by the package that carries
    it, XLA's target CPU and vector features, and the C library."""

    numpy: str
    openblas: str
    xla: str
    libc: str


def _numpy() -> str:
    found = opt_func_info(func_name="^(" + "|".join(KERNELS) + ")$", signature="float64")
    return " ".join(f"{name}={found[name]['dd']['current']}" for name in KERNELS)


def _openblas() -> str:
    """Each loaded OpenBLAS's core, asked of the library as threadpoolctl asks it."""
    maps = Path("/proc/self/maps")
    if not maps.exists():
        return "unknown"
    paths = sorted(
        {
            Path(line.split()[-1])
            for line in maps.read_text().splitlines()
            if "openblas" in line.rsplit("/", 1)[-1]
        }
    )
    cores = []
    for path in paths:
        library = ctypes.CDLL(str(path))
        for symbol in _CORENAME:
            corename = getattr(library, symbol, None)
            if corename is not None:
                corename.restype = ctypes.c_char_p
                cores.append(f"{path.parent.name.removesuffix('.libs')}={corename().decode()}")
                break
    return " ".join(cores) or "none"


def _fields(message: bytes) -> Iterator[tuple[int, bytes]]:
    """The length-delimited fields of a serialised protobuf message, in order."""
    i = 0
    while i < len(message):
        tag, i = _varint(message, i)
        number, wire = tag >> 3, tag & 7
        if wire == 0:
            _, i = _varint(message, i)
        elif wire == 1:
            i += 8
        elif wire == 5:
            i += 4
        elif wire == 2:
            size, i = _varint(message, i)
            yield number, message[i : i + size]
            i += size
        else:
            raise ValueError(f"wire type {wire} in a CPU topology")


def _varint(message: bytes, i: int) -> tuple[int, int]:
    value = shift = 0
    while True:
        byte = message[i]
        i += 1
        value |= (byte & 0x7F) << shift
        shift += 7
        if byte < 0x80:
            return value, i


def _field(message: bytes, number: int) -> bytes:
    return next((value for n, value in _fields(message) if n == number), b"")


def _xla() -> str:
    """The CPU and vector features XLA compiles for, read from its CPU topology: a
    `PjRtTopologyDescriptionProto` whose field 9 is an `Any` holding a `CpuTopologyProto`, whose
    field 5 is the `TargetMachineOptionsProto`, the target's triple, CPU and features. The CPU
    follows `--xla_cpu_max_isa` (`sandybridge` under `AVX` on a Zen 3), as the features do."""
    topology = xla_client.get_topology_for_devices(jax.devices("cpu")).serialize()
    target = _field(_field(_field(topology, 9), 2), 5)
    cpu = _field(target, 2).decode()
    if not cpu:
        return "unknown"
    features = _field(target, 3).decode().split(",")
    simd = sorted(f[1:] for f in features if f.startswith("+") and f[1:].startswith(_SIMD))
    return " ".join([cpu, *simd])


@functools.cache
def key() -> Dispatch:
    """This machine's dispatch."""
    name, version = platform.libc_ver()
    return Dispatch(_numpy(), _openblas(), _xla(), f"{name} {version}".strip() or "unknown")


if __name__ == "__main__":
    print(repr(key()))
