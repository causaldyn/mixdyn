"""Simulation-based calibration (Talts, Betancourt, Simpson, Vehtari and Gelman 2018) of mixdyn's
Bayesian media-mix model: draw the parameters from the prior, the data from the model, the
posterior from the data, and read where the drawn parameters rank among the posterior's draws. A
posterior computed right puts every rank uniform.

**The design**, the Stan oracle's world (``stan_bayes_mmm.py``) named, ``normal``, ``student``,
``tanh`` or ``lognormal``: its panel's spend, controls and calendar, its three geo tests' spend, not
its lift estimates, and the model of the oracle, fitted on its own scale: the KPI's unit and each
channel's spend unit stay the world's, so the prior the simulation draws from is the prior the fit
reads. Simulation ``i``, from ``SEED`` and ``i``:

* every coordinate from its prior, in ``BayesModel.coordinates``' order and on its transform, drawn
  again while the mean is not above 0 in some observed week under log-normal noise, where such a
  point has no likelihood: the prior the posterior reads is the prior truncated to the support;
* the KPI on the observed weeks from the model's mean at those parameters and its noise, normal,
  Student-t or log-normal about the mean, and each test's gap over its readout from the channel's
  coefficient times the gap in saturated spend and the tests' noise;
* the posterior from two chains of ``DRAWS`` draws after ``WARMUP`` of warm-up, each started
  uniform in ``[-2, 2]`` and moved into the support as mixdyn moves its starts, the compiled
  sampler's;
* each coordinate's rank: how many of ``RANKED`` draws, every fifth of the chains' in order, lie
  below the drawn value.

**Pre-registered, 10-02, before any simulation ran**, the ``tanh`` world added the same day, before
any simulation ran either. ``SIMULATIONS`` simulations of each world. For each coordinate, a
chi-square test of its ranks' uniformity over ``BINS`` bins of equal width; the ranks are uniform
if no coordinate's p-value is below 0.05 over the number of coordinates (Bonferroni). Every
simulation counts, divergent or not; its divergences, largest R-hat and least bulk effective size
are recorded beside its ranks.

**Pre-registered, 10-03, before any simulation of it ran**: the ``lognormal`` world, with the same
test, the same counts and the same rule that every simulation counts; how many prior draws each
simulation drew again is recorded beside its ranks.

    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 uv run --project <mixdyn checkout> \\
        --no-sync python evidence/sbc_bayes_mmm.py run <world> --out <json> [--workers 4]
    uv run --project <mixdyn checkout> --no-sync python evidence/sbc_bayes_mmm.py report <json>
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import math
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import chi2
from stan_bayes_mmm import GEO, WORLDS, world

import mixdyn
from mixdyn.inference import compiled
from mixdyn.inference.diagnostics import diagnose
from mixdyn.models.bayes_mmm import (
    BayesModel,
    HillCurve,
    LogNormal,
    StudentT,
    _uniform,
    bayes_model,
)

jax.config.update("jax_enable_x64", True)

SEED, SIMULATIONS, CHAINS, WARMUP, DRAWS = 20261003, 500, 2, 500, 500
RANKED, BINS = 199, 20
"""Every fifth of the chains' 1000 draws, the last dropped, so that a rank takes 200 values, ten to
a bin."""


@functools.cache
def design(name: str) -> BayesModel:
    panel, tests, _, spec = world(name)
    return bayes_model(panel, GEO, spec, tests=tests)


def prior(model: BayesModel, rng: np.random.Generator) -> np.ndarray:
    """One point of the sampling coordinates drawn from the model's prior."""
    spec, m = model.spec, len(model.spec.media)
    controls = []
    for c in spec.controls:
        size = abs(rng.standard_normal())
        controls.append(
            math.log(size) if c.lower == 0.0 or c.upper == 0.0 else rng.standard_normal()
        )
    retention = rng.beta(1.0, 3.0, m)
    blocks = [
        rng.normal([b.roi.mu for b in spec.media], [b.roi.sigma for b in spec.media]),
        np.log(retention) - np.log1p(-retention),
        rng.normal(0.0, 1.0, m),
    ]
    if isinstance(spec.saturation, HillCurve):
        blocks.append(rng.normal(0.0, 0.5, m))
    blocks += [
        [rng.normal(1.0, 1.0)],
        rng.standard_normal(model.calendar.shape[1]),
        controls,
        [math.log(abs(rng.standard_normal()))],
    ]
    if isinstance(spec.likelihood, StudentT):
        blocks.append([math.log(rng.gamma(2.0, 10.0))])
    if model.arms:
        blocks.append([math.log(abs(rng.standard_normal()))])
    return np.concatenate([np.asarray(b, dtype=float) for b in blocks])


def simulated(model: BayesModel, theta: np.ndarray, rng: np.random.Generator) -> BayesModel:
    """The model with its KPI and its tests' gaps drawn from itself at ``theta``."""
    p, _ = model._constrained(jnp.asarray(theta))
    saturated, beta = model._terms(p)
    mean = np.asarray(
        p["intercept"][0]
        + model.calendar @ p["calendar"]
        + model.controls @ p["control"]
        + saturated @ beta
    )[model.observed]
    sigma = float(p["sigma"][0])
    if isinstance(model.spec.likelihood, StudentT):
        noise = rng.standard_t(float(p["nu"][0]), mean.size)
    else:
        noise = rng.standard_normal(mean.size)
    arms = tuple(
        dataclasses.replace(
            arms,
            gap=float(beta[arms.channel]) * np.asarray(model._tested(p, arms))
            + float(p["test_sigma"][0]) * rng.standard_normal(arms.gap.size),
        )
        for arms in model.arms
    )
    if isinstance(model.spec.likelihood, LogNormal):
        y = mean * np.exp(sigma * noise - sigma**2 / 2.0)
    else:
        y = mean + sigma * noise
    return dataclasses.replace(model, y=y, arms=arms)


def supported(model: BayesModel, theta: np.ndarray) -> bool:
    """Whether the model's likelihood is positive anywhere at ``theta``: under log-normal noise only
    where the mean is above 0 in every observed week."""
    if not isinstance(model.spec.likelihood, LogNormal):
        return True
    p, _ = model._constrained(jnp.asarray(theta))
    return bool(np.min(model._mean(p, *model._terms(p))) > 0.0)


def simulation(job: tuple[str, int]) -> dict[str, Any]:
    name, i = job
    model = design(name)
    rng = np.random.default_rng([SEED, list(WORLDS).index(name), i])
    theta, redrawn = prior(model, rng), 0
    while not supported(model, theta):
        theta, redrawn = prior(model, rng), redrawn + 1
    fitted = simulated(model, theta, rng)
    start = time.perf_counter()
    chains = compiled.sample(
        fitted._log_density,
        _uniform(fitted, rng, CHAINS),
        seed=int(rng.integers(2**31)),
        warmup=WARMUP,
        draws=DRAWS,
    )
    seconds = time.perf_counter() - start
    kept = chains.positions.reshape(-1, model.dimension)[::5][:RANKED]
    health = diagnose(chains.positions)
    out = {
        "simulation": i,
        "theta": theta.tolist(),
        "ranks": (kept < theta).sum(axis=0).tolist(),
        "divergent": int(chains.divergent.sum()),
        "rhat": float(np.max(health.rhat)),
        "ess_bulk": float(np.min(health.ess_bulk)),
        "seconds": seconds,
        "redrawn": redrawn,
    }
    print(json.dumps({k: out[k] for k in ("simulation", "divergent", "rhat")}), flush=True)
    return out


def checkout() -> dict[str, Any]:
    """The mixdyn checkout the simulations ran on: its path, commit and whether it had changes."""
    root = Path(mixdyn.__file__).resolve().parents[2]

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, check=True
        ).stdout.strip()

    return {
        "path": str(root),
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
    }


def run(name: str, out: Path, workers: int) -> None:
    """Every simulation not yet in ``out``'s ``.jsonl``, then ``out``."""
    source = checkout()
    partial = out.with_suffix(".jsonl")
    done = {}
    if partial.exists():
        for line in partial.read_text().splitlines():
            record = json.loads(line)
            done[record["simulation"]] = record
    jobs = [(name, i) for i in range(SIMULATIONS) if i not in done]
    with ProcessPoolExecutor(workers) as pool, partial.open("a") as sink:
        for future in as_completed([pool.submit(simulation, job) for job in jobs]):
            record = future.result()
            sink.write(json.dumps(record) + "\n")
            sink.flush()
            done[record["simulation"]] = record
    out.write_text(
        json.dumps(
            {
                "world": name,
                "mixdyn": source,
                "seed": SEED,
                "chains": CHAINS,
                "warmup": WARMUP,
                "draws": DRAWS,
                "ranked": RANKED,
                "bins": BINS,
                "coordinates": list(design(name).coordinates),
                "simulations": [done[i] for i in sorted(done)],
            },
            indent=1,
        )
    )


def report(record: dict[str, Any]) -> None:
    ranks = np.array([s["ranks"] for s in record["simulations"]])
    n, dimension = ranks.shape
    counts = np.stack(
        [
            np.bincount(
                ranks[:, k] * record["bins"] // (record["ranked"] + 1), minlength=record["bins"]
            )
            for k in range(dimension)
        ]
    )
    expected = n / record["bins"]
    statistic = ((counts - expected) ** 2 / expected).sum(axis=1)
    p = chi2.sf(statistic, record["bins"] - 1)
    level = 0.05 / dimension
    print(
        f"{record['world']}: {n} simulations on mixdyn {record['mixdyn']['commit'][:7]}, level"
        f" {level:.2e} a coordinate"
    )
    for name, value, row in zip(record["coordinates"], p, counts, strict=True):
        print(f"  {name:28s} p {value:.4f}{'  REJECTED' if value < level else ''}  {row.tolist()}")
    divergent = [s["simulation"] for s in record["simulations"] if s["divergent"]]
    print(
        f"  uniform: {'yes' if (p >= level).all() else 'no'}; divergent in {len(divergent)}"
        f" simulations; largest R-hat {max(s['rhat'] for s in record['simulations']):.3f}; least"
        f" bulk ESS {min(s['ess_bulk'] for s in record['simulations']):.0f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    go = sub.add_parser("run")
    go.add_argument("world", choices=tuple(WORLDS))
    go.add_argument("--out", type=Path, required=True)
    go.add_argument("--workers", type=int, default=4)
    show = sub.add_parser("report")
    show.add_argument("record", type=Path)
    args = parser.parse_args()
    if args.command == "report":
        report(json.loads(args.record.read_text()))
        return
    run(args.world, args.out, args.workers)


if __name__ == "__main__":
    main()
