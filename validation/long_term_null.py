"""The nested model's long-term share on worlds with no long-term effect, and a dual adstock's on
one of them (ADR 0014, amended 2026-11-06).

A long-term null world is 104 weeks of one geo in which nothing television builds lasts in sales:
the case Cain (2025, Applied Marketing Analytics 11(1)) asks a model to rule out before it reads a
long-term effect. Television moves sales directly by 1 a euro, carried over at 0.2; search ads by
1.5 a euro, carried over at 0.1; price by -20 a euro. Branded searches, the brand measure, and
sales share a season of one harmonic; the searches' noise is 15 a week and sales' 20. Each curve is
a Hill of slope 1 at the channel's mean positive spend, as in ADR 0014's test world. The path from
television through the searches to sales is cut at one of its two links:

- `"unread"`: television builds the searches, 0.5 a euro carried over at 0.7, as in ADR 0014's
  test world, and sales do not read them;
- `"unbuilt"`: sales read the searches, 2 a search, and television does not move them.

Both worlds of a seed draw the same spend, price and noise.

Two readings of television's long-term share, at each draw of the posterior:

- the nested model (`mixdyn.models.nested`): stage one the searches on television, its return
  prior `LogNormal(0, 1)`, stage two sales on television, the search ads, price held at or below
  0 and the searches held at or above 0, both with the season. The share is the return through the
  searches over the total, `gamma r_1 / (gamma r_1 + r_2)`, `gamma` the searches' coefficient,
  `r_1` television's return on the searches and `r_2` its direct return on sales; it is read on
  both worlds;
- the same nested model with stage one's return prior half-normal, `HalfNormalRoi()` at the
  searches' scale (`mixdyn.models.nested`), whose density is greatest at 0: `nested half-normal`,
  read on both worlds where it is named;
- a dual adstock, the practice Cain finds reads spurious long-term effects and ADR 0014 does not
  ship: one stage of sales on television twice, once as spent and once as its spend carried over at
  0.9 by a normalised kernel, each with its own return, decay and curve, the search ads, price and
  the season. The share is the second's contribution over both; it is read on the `"unread"` world.

Each model is drawn by two chains of 300 warm-up iterations and 200 draws, from Pathfinder's
starts, the settings of the nested model's tests.

**What covering 0 reads.** The model's returns are above 0 and the searches' coefficient is
held above 0, so the share's posterior is on `(0, 1)` and a central interval never holds 0
exactly (ADR 0014). A share's 90 % interval is read as covering 0 where its lower end is below
`NIL`, 0.02: a fiftieth of the channel's return. `NIL` was set on a pilot of worlds of other seeds
before this harness ran: on three `"unread"` worlds the nested model's lower ends were 0.0006 to
0.0095 and the dual adstock's 0.070 to 0.154; on three `"unbuilt"` worlds the nested model's
were 0.057 to 0.092.

The seeds: `numpy.random.SeedSequence(ENTROPY)` spawns one child a world, and each world's child
two, the first the world's draws and the second the seed every model of the world is fitted from.
A reduced run of the first worlds reads the full run's first worlds.

Run `python -m validation.long_term_null` from the repository's root, JAX on the CPU, for the
table of `validation/long_term_null.md`. `--worlds` sets the `"unread"` worlds the nested model
reads, 100 by default; `--dual` the first of them the dual adstock reads, 20 by default; and
`--unbuilt` the `"unbuilt"` ones, 10 by default, the first of the same seeds. `--start` is the
first world read, 0 by default, so a run extends a record from where it stopped, and
`--tally FILE` prints the summary lines of the rows a record holds and reads no world. `--models`
names the models read, `"nested"` and `"dual adstock"` by default; the record's second table is
`--models "nested half-normal" --worlds 20 --unbuilt 20`.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy import stats

from mixdyn import Cadence, Channel, Geo, MediaPanel, Revenue
from mixdyn.models.bayes_mmm import (
    BayesFit,
    BayesMedia,
    BayesSpec,
    HalfNormalRoi,
    RoiPrior,
    fit_bayes,
)
from mixdyn.models.nested import NestedFit, NestedSpec, fit_nested
from mixdyn.models.ridge_mmm import Control, FourierSeason

type _Array = npt.NDArray[np.float64]
type Cut = Literal["unread", "unbuilt"]

ENTROPY = 20261106
WORLDS = 100
DUAL_WORLDS = 20
WEEKS = 104
UNBUILT = 10
MODELS = ("nested", "dual adstock")
"""The models a run reads by default; `"nested half-normal"` is read where it is named."""
NIL = 0.02
LEVEL = 0.9
LONG = 0.9
"""The retention a dual adstock's second column is carried over at."""
CHAINS, WARMUP, DRAWS = 2, 300, 200
NATIONAL = Geo("national")
SEASON = FourierSeason(52.18, 1)
NESTED = NestedSpec(
    "searches",
    BayesSpec((BayesMedia("tv", RoiPrior(0.0, 1.0)),), season=SEASON),
    BayesSpec(
        (BayesMedia("tv"), BayesMedia("search")),
        (Control("price", upper=0.0), Control("searches", lower=0.0)),
        season=SEASON,
    ),
)
HALF = dataclasses.replace(
    NESTED,
    stage_one=dataclasses.replace(NESTED.stage_one, media=(BayesMedia("tv", HalfNormalRoi()),)),
)
"""`NESTED` with television's return on the searches half-normal at the searches' scale."""
DUAL = BayesSpec(
    (BayesMedia("tv"), BayesMedia("tv_long"), BayesMedia("search")),
    (Control("price", upper=0.0),),
    season=SEASON,
)


def _saturated(spend: _Array, retention: float) -> _Array:
    """The spend carried over by a normalised geometric kernel from the first week, over its mean
    positive spend, through a Hill of slope 1 and half-saturation 1."""
    weights = retention ** np.arange(spend.size)
    adstock = np.convolve(spend / spend[spend > 0].mean(), weights / weights.sum())[: spend.size]
    return adstock / (adstock + 1.0)


def world(seed: np.random.SeedSequence, cut: Cut) -> MediaPanel:
    """One long-term null world, its path through the searches cut where `cut` says: sales,
    television's and the search ads' spend, price and the searches."""
    rng = np.random.default_rng(seed)
    t = np.arange(WEEKS)
    tv = rng.gamma(2.0, 50.0, WEEKS) * (rng.random(WEEKS) > 0.3)
    search = rng.gamma(2.0, 30.0, WEEKS) * (rng.random(WEEKS) > 0.2)
    season = np.sin(2.0 * np.pi * t / 52.18)
    h = _saturated(tv, 0.7)
    built = 0.5 if cut == "unread" else 0.0
    searches = (
        500.0 + 50.0 * season + built * tv.sum() * h / h.sum() + 15.0 * rng.standard_normal(WEEKS)
    )
    direct, paid = _saturated(tv, 0.2), _saturated(search, 0.1)
    price = 10.0 + 0.5 * rng.standard_normal(WEEKS)
    sales = (
        2000.0
        + 40.0 * season
        + tv.sum() * direct / direct.sum()
        + 1.5 * search.sum() * paid / paid.sum()
        + (2.0 if cut == "unbuilt" else 0.0) * searches
        - 20.0 * (price - 10.0)
        + 20.0 * rng.standard_normal(WEEKS)
    )
    return MediaPanel(
        dates=np.datetime64("2023-01-02") + np.timedelta64(7, "D") * t,
        cadence=Cadence.WEEK,
        geos=(NATIONAL,),
        channels=(Channel("tv"), Channel("search")),
        controls=("price", "searches"),
        kpi=sales[None, :],
        kpi_unit=Revenue(),
        spend=np.column_stack([tv, search])[None],
        currency="EUR",
        control_values=np.column_stack([price, searches])[None],
        filled=np.zeros((1, WEEKS), dtype=bool),
    )


def dual(panel: MediaPanel) -> MediaPanel:
    """`panel` with television twice, as spent and carried over at `LONG`, and price alone among
    its controls."""
    tv = panel.spend[0, :, 0]
    weights = LONG ** np.arange(WEEKS)
    carried = np.convolve(tv, weights / weights.sum())[:WEEKS]
    return dataclasses.replace(
        panel,
        channels=(Channel("tv"), Channel("tv_long"), Channel("search")),
        spend=np.column_stack([tv, carried, panel.spend[0, :, 1]])[None],
        controls=("price",),
        control_values=panel.control_values[:, :, :1],
    )


def nested_share(fit: NestedFit) -> _Array:
    """Television's share of its return on sales that runs through the searches, at each draw."""
    mediated = fit.mediator_coefficient() * fit.stage_one.draws["roi"][..., 0]
    return mediated / fit.roi()["tv"]


def dual_share(fit: BayesFit) -> _Array:
    """The carried column's share of television's two columns' contribution, at each draw."""
    spent = np.array([fit.spend[c][-WEEKS:].sum() for c in ("tv", "tv_long")])
    contribution = fit.draws["roi"][..., :2] * spent
    return contribution[..., 1] / contribution.sum(axis=-1)


@dataclass(frozen=True, slots=True)
class Reading:
    """One world's long-term share under one model: its 90 % interval and median, whether the
    interval covers 0 as the module reads it, and the fit's divergent transitions."""

    model: str
    cut: Cut
    world: int
    low: float
    median: float
    high: float
    divergent: int

    @property
    def covers(self) -> bool:
        return self.low < NIL


def _interval(share: _Array) -> tuple[float, float, float]:
    low, median, high = np.quantile(share, [(1.0 - LEVEL) / 2.0, 0.5, (1.0 + LEVEL) / 2.0])
    return float(low), float(median), float(high)


def seeds(worlds: int) -> list[tuple[np.random.SeedSequence, int]]:
    """Each world's draws and the integer seed its models are fitted from."""
    out = []
    for child in np.random.SeedSequence(ENTROPY).spawn(worlds):
        data, fits = child.spawn(2)
        out.append((data, int(fits.generate_state(1)[0])))
    return out


def read(
    k: int,
    data: np.random.SeedSequence,
    seed: int,
    cut: Cut,
    *,
    dual_adstock: bool = True,
    nested: Sequence[str] = ("nested",),
) -> list[Reading]:
    """World `k`'s long-term share, its path cut at `cut`, under each nested model `nested` names,
    `"nested"` and `"nested half-normal"`, its stage one's return prior half-normal, and, on the
    `"unread"` world where `dual_adstock` says, under the dual adstock."""
    panel = world(data, cut)
    out = []
    for name, spec in (("nested", NESTED), ("nested half-normal", HALF)):
        if name not in nested:
            continue
        fit = fit_nested(
            panel, NATIONAL, spec, seed=seed, chains=CHAINS, warmup=WARMUP, draws=DRAWS
        )
        divergent = int(fit.stage_one.divergent.sum() + fit.stage_two.divergent.sum())
        out.append(Reading(name, cut, k, *_interval(nested_share(fit)), divergent))
    if cut == "unread" and dual_adstock:
        two = fit_bayes(
            dual(panel), NATIONAL, DUAL, seed=seed, chains=CHAINS, warmup=WARMUP, draws=DRAWS
        )
        share = _interval(dual_share(two))
        out.append(Reading("dual adstock", cut, k, *share, int(two.divergent.sum())))
    return out


def interval(covered: int, worlds: int, level: float = 0.95) -> tuple[float, float]:
    """The Clopper-Pearson interval at `level` of a rate read as `covered` of `worlds`."""
    tail = (1.0 - level) / 2.0
    low = 0.0 if covered == 0 else float(stats.beta.ppf(tail, covered, worlds - covered + 1))
    high = (
        1.0 if covered == worlds else float(stats.beta.ppf(1 - tail, covered + 1, worlds - covered))
    )
    return low, high


ROW = re.compile(
    r"^\| (nested half-normal|nested|dual adstock) \| (unread|unbuilt) \| (\d+) \| ([\d.]+) \| "
    r"([\d.]+) \| ([\d.]+) \| (?:yes|no) \| (\d+) \|$",
    re.MULTILINE,
)
"""A record's table row: model, cut, world, low, median, high, whether it covers 0, divergent."""


def row(r: Reading) -> str:
    """`r` as a row of the record's table."""
    return (
        f"| {r.model} | {r.cut} | {r.world} | {r.low:.4f} | {r.median:.4f} | {r.high:.4f} | "
        f"{'yes' if r.covers else 'no'} | {r.divergent} |"
    )


def rows(text: str) -> list[Reading]:
    """The readings a record's table holds, read back from its rows."""
    return [
        Reading(model, cut, int(k), float(low), float(median), float(high), int(divergent))
        for model, cut, k, low, median, high, divergent in ROW.findall(text)
    ]


def summary(readings: list[Reading]) -> list[str]:
    """For each model and cut the readings hold: the worlds whose interval covers 0, their rate,
    its Monte Carlo standard error and its 95 % Clopper-Pearson interval."""
    out = []
    for model, cut in (
        ("nested", "unread"),
        ("dual adstock", "unread"),
        ("nested", "unbuilt"),
        ("nested half-normal", "unread"),
        ("nested half-normal", "unbuilt"),
    ):
        mine = [r for r in readings if r.model == model and r.cut == cut]
        if not mine:
            continue
        covered = sum(r.covers for r in mine)
        low, high = interval(covered, len(mine))
        se = math.sqrt(covered * (len(mine) - covered) / len(mine) ** 3)
        out.append(
            f"- {model}, {cut}: covers 0 in {covered} of {len(mine)} worlds, "
            f"{covered / len(mine):.2f} (SE {se:.3f}; 95 % Clopper-Pearson [{low:.3f}, {high:.3f}])"
        )
    return out


def main(argv: list[str] | None = None) -> None:
    import jax

    jax.config.update("jax_enable_x64", True)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--worlds", type=int, default=WORLDS)
    parser.add_argument("--dual", type=int, default=DUAL_WORLDS)
    parser.add_argument("--unbuilt", type=int, default=UNBUILT)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--tally", type=Path)
    parser.add_argument("--models", default=",".join(MODELS))
    args = parser.parse_args(argv)
    models = args.models.split(",")
    unknown = sorted(set(models) - {*MODELS, "nested half-normal"})
    if unknown:
        parser.error(f"no model {unknown}")
    nested = [m for m in models if m != "dual adstock"]
    if args.tally is not None:
        print("\n".join(summary(rows(args.tally.read_text(encoding="utf-8")))))
        return
    readings: list[Reading] = []
    print("| model | cut | world | low | median | high | covers 0 | divergent |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for k, (data, seed) in enumerate(seeds(max(args.worlds, args.unbuilt))):
        if k < args.start:
            continue
        cuts: list[Cut] = ["unread"] if k < args.worlds else []
        cuts += ["unbuilt"] if k < args.unbuilt else []
        for cut in cuts:
            dual_adstock = k < args.dual and "dual adstock" in models
            for r in read(k, data, seed, cut, dual_adstock=dual_adstock, nested=nested):
                readings.append(r)
                print(row(r), flush=True)
    print()
    print("\n".join(summary(readings)))


if __name__ == "__main__":
    main()
