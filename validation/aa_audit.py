"""Every experiment reader read on null panels: the share of tests that reject where the test
changed nothing, and the share of intervals that hold 0 (ADR 0058).

A null panel is 104 weeks of a KPI in 10 or 30 geos with no treatment effect. Each geo `g` has a
level `mu_g`, 100 times a log-normal of sd 0.5, and a seasonality all geos share, `1 + 0.1 sin(2 pi
t / 52)`; its noise is `0.05 mu_g` times a unit series of one of three laws:

- "ar1": a stationary first-order autoregression of coefficient 0.5 and variance 1;
- "trend": the same, plus a linear trend of its own a geo, `b_g t / 52`, `b_g` a standard normal,
  so that a geo's drift over a year is of the order of its noise;
- "shift": the same, plus a level shift of 2 all geos share, from a week drawn uniformly among the
  panel's 2nd to its last.

Each reader reads the panel at its shipped defaults on a null treatment its own design draws:

- the geo readers, `read_did`, `read_tbr` and `read_synthetic_control`, on a design randomised
  within matched pairs, `assign(pair_geos(mean KPI over the pre-period))`, the test over the last 8
  weeks, the 96 before them its pre-period;
- `read_sdid` on 30 % of the geos, rounded half up, drawn at random as treated over the same last
  8 weeks, every other geo a donor;
- `read_switchback`, at each of its methods, on the KPI summed over geos, a `switchback_design` of
  the 104 weeks in 26 blocks of 4 with a washout of 1, half the blocks on.

The seeds: `numpy.random.SeedSequence(ENTROPY)` spawns one child a panel cell, a noise law and a
geo count, in `CELLS`' order; the cell's child spawns one a draw, from which the panel's noise is
drawn and the design's integer seed taken. Every reader reads the same panels, so a reader and its
alternative are compared draw by draw, and a reduced run of the first draws reads the full run's
first panels.

A cell's false-positive rate is the share of draws whose p-value is below 0.05, its Monte Carlo
standard error `sqrt(0.05 0.95 / draws)`, 0.0154 at 200; the coverage is the share whose 95 %
interval holds 0. A cell passes where the rate is within `BAND`, [0.03, 0.08]. Its top is 0.05
plus two standard errors, 0.0808, rounded to the hundredth. Its floor is set tighter than the
0.0192 two standard errors give, so that a reader that rejects at well under its level, and so
spends power, fails too. At 200 draws a cell passes on 6 to 16 rejections: a reader at the nominal
level passes with probability 0.914, so about 3 of 36 such cells fail by chance; one at 0.10,
twice nominal, passes with probability 0.207, and one at 0.025 with 0.384. A reader that refuses
a draw (`ValueError`) is counted apart and the rates are read over the draws it read.

Run `uv run --frozen python -m validation.aa_audit` from the repository's root for the table, as
the markdown rows of `validation/aa_audit.md`; `--draws` and `--jobs` set the draws a cell and the
processes.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Literal, get_args

import numpy as np
import numpy.typing as npt

from mixdyn import Cadence, Channel, Geo, MediaPanel, Revenue
from mixdyn.experiments import (
    GeoReading,
    GeoTest,
    assign,
    pair_geos,
    read_did,
    read_pairs,
    read_sdid,
    read_switchback,
    read_synthetic_control,
    read_tbr,
    switchback_design,
)
from mixdyn.experiments.sdid import Variance
from mixdyn.experiments.switchback import Method

type _Array = npt.NDArray[np.float64]
type Noise = Literal["ar1", "trend", "shift"]

ENTROPY = 20261031
NOISES: tuple[Noise, ...] = ("ar1", "trend", "shift")
GEOS = (10, 30)
CELLS = tuple((noise, geos) for noise in NOISES for geos in GEOS)
WEEKS, READOUT = 104, 8
RHO, SCALE, SEASON, SHIFT = 0.5, 0.05, 0.1, 2.0
SDID_SHARE = 0.3
BLOCK, WASHOUT = 4, 1
DRAWS, ALPHA = 200, 0.05
BAND = (0.03, 0.08)

CHANNEL = Channel("test")
DATES = np.datetime64("2024-01-01") + np.timedelta64(7, "D") * np.arange(WEEKS)


@dataclass(frozen=True, slots=True)
class Read:
    """One reading's p-value against no effect and its interval's ends."""

    p_value: float
    lower: float
    upper: float


@dataclass(frozen=True, slots=True)
class Cell:
    """A reader's cell: the share of the `read` draws that rejected at 0.05, `fpr`, its Monte Carlo
    standard error `se`, the share whose interval held 0, `coverage`, and the draws `refused`."""

    reader: str
    noise: Noise
    geos: int
    read: int
    refused: int
    fpr: float
    se: float
    coverage: float

    @property
    def passes(self) -> bool:
        return BAND[0] <= self.fpr <= BAND[1]


def null_panel(noise: Noise, geos: int, rng: np.random.Generator) -> _Array:
    """A KPI of `geos` geos by `WEEKS` weeks with no treatment effect, under `noise`; see the
    module."""
    level = 100.0 * np.exp(0.5 * rng.standard_normal(geos))
    t = np.arange(WEEKS)
    season = 1.0 + SEASON * np.sin(2.0 * math.pi * t / 52.0)
    shocks = rng.standard_normal((geos, WEEKS))
    u = np.empty((geos, WEEKS))
    u[:, 0] = shocks[:, 0]
    innovation = math.sqrt(1.0 - RHO * RHO)
    for k in range(1, WEEKS):
        u[:, k] = RHO * u[:, k - 1] + innovation * shocks[:, k]
    if noise == "trend":
        u += rng.standard_normal(geos)[:, None] * t / 52.0
    elif noise == "shift":
        u += SHIFT * (t >= rng.integers(1, WEEKS))
    elif noise != "ar1":
        msg = f"noise must be one of {NOISES}, got {noise!r}"
        raise ValueError(msg)
    return level[:, None] * (season + SCALE * u)


def _panel(kpi: _Array) -> MediaPanel:
    geos, weeks = kpi.shape
    return MediaPanel(
        dates=DATES[:weeks],
        cadence=Cadence.WEEK,
        geos=tuple(Geo(f"g{g}") for g in range(geos)),
        channels=(CHANNEL,),
        controls=(),
        kpi=kpi,
        kpi_unit=Revenue(),
        spend=np.zeros((geos, weeks, 1)),
        currency="EUR",
        control_values=np.zeros((geos, weeks, 0)),
        filled=np.zeros((geos, weeks), dtype=bool),
    )


_PRE = WEEKS - READOUT
_SPEND = np.ones(WEEKS)
_PLANNED = np.concatenate([np.ones(_PRE), np.zeros(READOUT)])


def _paired(kpi: _Array, seed: int) -> tuple[GeoTest, list[tuple[Geo, Geo]]]:
    """The matched-pair design's null test over the last `READOUT` weeks, and its pairs."""
    blocks = pair_geos(kpi[:, :_PRE].mean(axis=1))
    design = assign(blocks, seed)
    test = GeoTest(
        CHANNEL,
        tuple(Geo(f"g{g}") for g in design.treated),
        tuple(Geo(f"g{g}") for g in design.control),
        DATES[_PRE],
        DATES[-1],
        _SPEND,
        _PLANNED,
        READOUT,
    )
    return test, [(Geo(f"g{a}"), Geo(f"g{b}")) for a, b in blocks.blocks]


def _geo(
    reader: Callable[..., tuple[object, GeoReading]], *, pairs: bool = False, **options: object
) -> Reader:
    def read(kpi: _Array, seed: int) -> Read:
        test, matched = _paired(kpi, seed)
        if pairs:
            options["pairs"] = matched
        _, reading = reader(_panel(kpi), test, **options)
        return Read(reading.p_value, reading.lower, reading.upper)

    return read


def _sdid(variance: Variance = "placebo") -> Reader:
    def read(kpi: _Array, seed: int) -> Read:
        geos = kpi.shape[0]
        rng = np.random.default_rng(seed)
        count = math.floor(SDID_SHARE * geos + 0.5)
        treated = [Geo(f"g{g}") for g in sorted(rng.choice(geos, count, replace=False))]
        _, reading = read_sdid(
            _panel(kpi),
            treated,
            DATES[_PRE],
            DATES[-1],
            spend=_SPEND,
            planned=_PLANNED,
            seed=seed,
            variance=variance,
        )
        return Read(reading.p_value, reading.lower, reading.upper)

    return read


def _switchback_reader(method: Method) -> Reader:
    def read(kpi: _Array, seed: int) -> Read:
        design = switchback_design(DATES, BLOCK, WASHOUT, seed=seed)
        spend = design.assignment.astype(np.float64)
        _, reading = read_switchback(kpi, spend, design, method=method)
        return Read(reading.p_value, reading.lower, reading.upper)

    return read


type Reader = Callable[[_Array, int], Read]

READERS: dict[str, Reader] = {
    "did": _geo(read_did),
    "tbr": _geo(read_tbr),
    "synthetic_control": _geo(read_synthetic_control),
    "sdid": _sdid(),
    # The opt-in alternatives of the readers that fail a cell at their defaults.
    "did:unit_placebo": _geo(read_did, pairs=True, variance="unit_placebo"),
    "tbr:unit_placebo": _geo(read_tbr, pairs=True, variance="unit_placebo"),
    "pairs:sign_flip": _geo(read_pairs, pairs=True),
    "sdid:unit_placebo": _sdid(variance="unit_placebo"),
    **{f"switchback:{m}": _switchback_reader(m) for m in get_args(Method.__value__)},
}


def _draws(noise: Noise, geos: int, draws: int) -> list[tuple[_Array, int]]:
    """The cell's panels and the designs' seeds, one a draw."""
    cell = np.random.SeedSequence(ENTROPY).spawn(len(CELLS))[CELLS.index((noise, geos))]
    out = []
    for child in cell.spawn(draws):
        seed = int(child.generate_state(1)[0])
        out.append((null_panel(noise, geos, np.random.default_rng(child)), seed))
    return out


def audit(reader: str, noise: Noise, geos: int, draws: int = DRAWS) -> Cell:
    """`reader`'s cell under `noise` at `geos` geos over `draws` null panels."""
    read = READERS[reader]
    rejected = held = refused = 0
    for kpi, seed in _draws(noise, geos, draws):
        try:
            result = read(kpi, seed)
        except ValueError:
            refused += 1
            continue
        rejected += result.p_value < ALPHA
        held += result.lower <= 0.0 <= result.upper
    count = draws - refused
    fpr = rejected / count if count else math.nan
    coverage = held / count if count else math.nan
    se = math.sqrt(ALPHA * (1.0 - ALPHA) / count) if count else math.nan
    return Cell(reader, noise, geos, count, refused, fpr, se, coverage)


def _row(cell: Cell) -> str:
    verdict = "pass" if cell.passes else "FAIL"
    refused = f" ({cell.refused} refused)" if cell.refused else ""
    return (
        f"| {cell.reader} | {cell.noise} | {cell.geos} | {cell.fpr:.3f} | {cell.se:.4f} | "
        f"{cell.coverage:.3f} | {verdict}{refused} |"
    )


def _audit(job: tuple[str, Noise, int, int]) -> Cell:
    return audit(*job)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--draws", type=int, default=DRAWS)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--readers", nargs="*", default=list(READERS))
    args = parser.parse_args(argv)
    jobs = [(r, n, g, args.draws) for r in args.readers for n, g in CELLS]
    with ProcessPoolExecutor(args.jobs) as pool:
        cells = list(pool.map(_audit, jobs))
    print("| reader | noise | geos | FPR | SE | coverage | band |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for cell in cells:
        print(_row(cell))


if __name__ == "__main__":
    main()
