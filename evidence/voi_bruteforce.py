"""Records the value of lift tests on a tiny world by refitting, as mixdyn's oracle fixture
tests/fixtures/oracles/voi_bruteforce.json: each affordable set of four tests valued by
``choose_tests`` under ``Refit``, its posterior refitted by CmdStan at every outer draw.

**The world.** Two channels of ``tanh`` curves, 104 weeks and normal noise, with no control, trend
or season: the smallest spec the model accepts. Its spend is gamma, a week in twenty dark, and its
KPI is drawn from the model at ``TRUTH``, the same return on each channel by curves apart. The plan
spends the channels' mean spend a week over the next ``PERIODS`` weeks, each channel within a
quarter and twice its mean spend a week. So the plan without a test sits inside its ranges and its
split is uncertain, which a test's value needs: where the plan sits at a corner, as with returns of
1.5 and 3, no test moves it and every value is 0.

**The tests**, ``designs``' after the panel at the plan in place: each channel's go-dark and its
doubling, eight weeks and a cooldown of eight read, in a quarter of the market, their standard
errors ``GroupNoise(NOISE)``'s. Two take half the market and three more, so the affordable sets are
the four and their six pairs.

**The posterior** is CmdStan's, on ``stan_bayes_mmm.py``'s program, an independent sampler and
model code: ``CHAINS`` chains of ``DRAWS`` draws after ``WARMUP`` of warm-up, mapped to mixdyn's
coordinates by that harness's inverse transforms and to the panel's units by mixdyn's own map. The
readings keep ``KEPT`` draws, evenly spaced over each chain.

**The refits.** ``choose_tests(..., draws=OUTER, update=Refit(OUTER // 2, fit))`` refits every
outer draw, so each set's reduction is its refits' alone, beside its ``Reweight`` reductions at the
same draws with the same normals. ``fit`` is CmdStan on the world with the set's simulated
estimates, as the base posterior, from a seed read off the estimates. The estimates a call asks for
do not depend on what the refits return, so a first pass records them with a fit that returns two
of the base posterior's draws, cheap to plan, its values discarded; the refits run in parallel, each
cached on disk by its estimates, and a second pass values the sets from the cache.

**The fixture** holds the world, the tests and their design, the plan, the base posterior's kept
draws in the panel's units, and for each set its refitted and reweighted reductions at each outer
draw; each refit's divergences and largest R-hat.

CmdStan comes from the author's install, never a dependency:

    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 timeout 43200 uv run --project \\
        <mixdyn checkout> --no-sync python evidence/voi_bruteforce.py <cmdstan home> \\
        <work directory> <fixture path> [--workers 4]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import jax
import numpy as np
from stan_bayes_mmm import PROGRAM, csv, data, saturated, stan, unconstrained

from mixdyn import Cadence, Channel, Geo, MediaPanel, Revenue
from mixdyn.calibration import LiftEstimate
from mixdyn.decisions import Range, Readings
from mixdyn.experiments import (
    After,
    Candidate,
    ExperimentChoice,
    GroupNoise,
    Refit,
    choose_tests,
    designs,
)
from mixdyn.inference.diagnostics import diagnose
from mixdyn.models.bayes_mmm import BayesMedia, BayesSpec, Normal, TanhCurve, _summary, bayes_model

jax.config.update("jax_enable_x64", True)

WEEKS, SEED, CHAINS, WARMUP, DRAWS, KEPT, OUTER = 104, 20261005, 4, 1000, 1000, 1000, 200
CHANNELS = ("tv", "search")
TRUTH = {"roi": [2.0, 2.0], "retention": [0.6, 0.1], "half": [1.0, 0.7]}
NOISE, PERIODS, LIMITS = 20.0, 13, (0.25, 2.0)
SPEND = (8.0, 10.0, 0.05)
MULTIPLIERS, DURATIONS, SHARES = (0.0, 2.0), (8,), (0.25,)
START = "2023-01-02"
GEO = Geo("national")


def world() -> tuple[MediaPanel, BayesSpec]:
    rng = np.random.default_rng(SEED)
    shape, scale, dark = SPEND
    spend = rng.gamma(shape, scale, (WEEKS, 2)) * (rng.random((WEEKS, 2)) >= dark)
    unit = np.array([s[s > 0].mean() for s in spend.T])
    columns = []
    for j in range(2):
        shape = (TRUTH["retention"][j], TRUTH["half"][j], 1.0, TanhCurve())
        h = saturated(spend[:, j] / unit[j], *shape)
        columns.append(TRUTH["roi"][j] * spend[:, j].sum() * h / h.sum())
    kpi = 1000.0 + np.sum(columns, axis=0) + NOISE * rng.standard_normal(WEEKS)
    panel = MediaPanel(
        dates=np.datetime64(START) + np.timedelta64(7, "D") * np.arange(WEEKS),
        cadence=Cadence.WEEK,
        geos=(GEO,),
        channels=tuple(Channel(c) for c in CHANNELS),
        controls=(),
        kpi=kpi[None, :],
        kpi_unit=Revenue(),
        spend=spend[None],
        currency="EUR",
        control_values=np.zeros((1, WEEKS, 0)),
        filled=np.zeros((1, WEEKS), dtype=bool),
    )
    spec = BayesSpec(
        tuple(BayesMedia(c) for c in CHANNELS), likelihood=Normal(), saturation=TanhCurve()
    )
    return panel, spec


def plan(panel: MediaPanel) -> tuple[float, dict[Channel, Range], dict[Channel, float]]:
    """The budget over the periods, each channel's range a week and its spend a week in place."""
    mean = panel.spend[0].mean(axis=0)
    least, most = LIMITS
    limits = {c: Range(least * m, most * m) for c, m in zip(panel.channels, mean, strict=True)}
    return (
        float(PERIODS * mean.sum()),
        limits,
        dict(zip(panel.channels, mean.tolist(), strict=True)),
    )


def tests(panel: MediaPanel) -> tuple[Candidate, ...]:
    _, _, current = plan(panel)
    return designs(
        panel,
        GEO,
        panel.channels,
        GroupNoise(NOISE),
        when=After(current),
        multipliers=MULTIPLIERS,
        durations=DURATIONS,
        shares=SHARES,
    )


def written(estimates: dict[Channel, list[LiftEstimate]]) -> list[object]:
    """The estimates as JSON reads them, by channel in their order."""
    return [
        [
            str(c),
            [
                [e.spend.tolist(), e.planned.tolist(), e.readout, e.lift, e.se, e.share]
                for e in estimates[c]
            ],
        ]
        for c in sorted(estimates)
    ]


def key(estimates: dict[Channel, list[LiftEstimate]]) -> str:
    """The fit's name: the world and the sampler's settings, with the estimates."""
    settings = [WEEKS, SEED, CHAINS, WARMUP, DRAWS, KEPT, TRUTH, NOISE, SPEND]
    return hashlib.sha256(json.dumps([settings, written(estimates)]).encode()).hexdigest()[:20]


def kept(chains: int, per_chain: int, count: int) -> np.ndarray:
    """``count`` draws evenly spaced over the chains' draws in order, from the first to the last."""
    return np.linspace(0, chains * per_chain - 1, count).round().astype(np.intp)


def posterior(
    executable: Path,
    work: Path,
    estimates: dict[Channel, list[LiftEstimate]],
    threads: int,
) -> dict[str, object]:
    """CmdStan's posterior of the world with ``estimates``, cached under their key: the kept
    draws in the panel's units, their chains, and the run's divergences and largest R-hat."""
    name = f"fit_{key(estimates)}"
    cached = work / f"{name}.json"
    if cached.exists():
        return json.loads(cached.read_text())
    panel, spec = world()
    model = bayes_model(panel, GEO, spec, estimates=estimates or None)
    (work / f"{name}.data.json").write_text(json.dumps(data(model, panel, {}, estimates)))
    seed = int(key(estimates)[:8], 16) if estimates else SEED
    stan(
        executable,
        work,
        name,
        ["sample", f"num_samples={DRAWS}", f"num_warmup={WARMUP}", f"num_chains={CHAINS}",
         "random", f"seed={seed}", f"num_threads={threads}"],
        "sample",
    )  # fmt: skip
    header, chains = None, []
    for c in range(1, CHAINS + 1):
        path = work / f"{name}.sample_{c}.csv"
        header, values = csv(path)
        chains.append(values)
        path.unlink()
    (work / f"{name}.data.json").unlink()
    draws = np.array(chains)
    theta = unconstrained(header, draws.reshape(-1, draws.shape[2]), model)
    raw, _, _ = _summary(model, theta)
    keep = kept(CHAINS, DRAWS, KEPT)
    shaped = {
        k: raw[k].reshape(CHAINS, DRAWS, -1)
        for k in ("retention", "half_saturation", "coefficient")
    }
    record = {
        **{k: raw[k][keep].tolist() for k in ("retention", "half_saturation", "coefficient")},
        "chain": (keep // DRAWS).tolist(),
        "length": int(model.lagged.shape[2]),
        "divergent": int(draws[:, :, header.index("divergent__")].sum()),
        "rhat": max(float(np.max(diagnose(v).rhat)) for v in shaped.values()),
    }
    cached.write_text(json.dumps(record))
    return record


def readings(record: dict[str, object], channels: tuple[Channel, ...]) -> Readings:
    retention = np.array(record["retention"])
    count = retention.shape[0]
    return Readings(
        channels=channels,
        retention=retention,
        half_saturation=np.array(record["half_saturation"]),
        slope=np.full_like(retention, np.nan),
        coefficient=np.array(record["coefficient"]),
        tanh=np.ones(count, dtype=bool),
        fit=np.zeros(count, dtype=np.intp),
        chain=np.array(record["chain"], dtype=np.intp),
        stacking=np.ones(1),
        pointwise=None,
        length=int(record["length"]),
    )


def choose(
    panel: MediaPanel, base: Readings, candidates: tuple[Candidate, ...], fit
) -> ExperimentChoice:
    budget, limits, _ = plan(panel)
    return choose_tests(
        panel,
        GEO,
        base,
        candidates,
        budget,
        PERIODS,
        limits,
        test_budget=1e12,
        seed=SEED,
        draws=OUTER,
        update=Refit(OUTER // 2, fit),
    )


def refit(arguments: tuple[Path, Path, dict[Channel, list[LiftEstimate]], int]) -> None:
    posterior(*arguments)
    # each refit's model compiles programs no later refit reuses; a worker that kept them grew
    # about 9 MiB a refit, and three of them filled a 6 GiB unit after 1050 refits
    jax.clear_caches()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cmdstan", type=Path)
    parser.add_argument("work", type=Path)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=1, help="CmdStan's threads a refit")
    arguments = parser.parse_args()
    cmdstan, work = arguments.cmdstan.resolve(), arguments.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    source = work / "bayes_mmm.stan"
    if not source.exists() or source.read_text() != PROGRAM:
        source.write_text(PROGRAM)
    executable = work / "bayes_mmm"
    subprocess.run(
        ["make", "-C", str(cmdstan), "STAN_THREADS=true", str(executable)],
        check=True,
        capture_output=True,
        timeout=1200,
    )
    panel, _ = world()
    candidates = tests(panel)
    base_record = posterior(executable, work, {}, max(arguments.threads, CHAINS))
    base = readings(base_record, panel.channels)

    asked: list[dict[Channel, list[LiftEstimate]]] = []

    few = readings(
        {k: v[:2] if k in ("retention", "half_saturation", "coefficient", "chain") else v
         for k, v in base_record.items()},
        panel.channels,
    )  # fmt: skip

    def record(estimates):
        asked.append({c: list(es) for c, es in estimates.items()})
        return few

    choose(panel, base, candidates, record)
    unique = {key(e): e for e in asked}
    print(f"{len(asked)} refits asked, {len(unique)} distinct", flush=True)
    with ProcessPoolExecutor(arguments.workers) as pool:
        jobs = [(executable, work, e, arguments.threads) for e in unique.values()]
        for done, _ in enumerate(pool.map(refit, jobs), start=1):
            if done % 50 == 0:
                print(f"{done} of {len(jobs)} refits", flush=True)

    records: list[dict[str, object]] = []

    def cached(estimates):
        records.append(posterior(executable, work, {c: list(es) for c, es in estimates.items()}, 1))
        return readings(records[-1], panel.channels)

    choice = choose(panel, base, candidates, cached)
    budget, limits, _ = plan(panel)
    version = (cmdstan / "makefile").read_text().split("CMDSTAN_VERSION := ")[1].split()[0]
    arguments.fixture.write_text(
        json.dumps(
            {
                "provenance": {
                    "generator": "evidence/voi_bruteforce.py: choose_tests under Refit, "
                    "every outer draw refitted by CmdStan on stan_bayes_mmm.py's program",
                    "cmdstan": version,
                    "seed": SEED,
                    "chains": CHAINS,
                    "warmup": WARMUP,
                    "draws": DRAWS,
                    "kept": KEPT,
                    "outer": OUTER,
                    "truth": TRUTH,
                    "noise": NOISE,
                    "spend": SPEND,
                },
                "world": {
                    "start": START,
                    "geo": str(GEO),
                    "channels": list(CHANNELS),
                    "kpi": panel.kpi[0].tolist(),
                    "spend": panel.spend[0].tolist(),
                },
                "plan": {
                    "budget": budget,
                    "periods": PERIODS,
                    "limits": [[r.least, r.most] for r in limits.values()],
                    "test_budget": 1e12,
                },
                "design": {
                    "multipliers": MULTIPLIERS,
                    "durations": DURATIONS,
                    "shares": SHARES,
                },
                "tests": [
                    {
                        "channel": str(c.channel),
                        "spend": c.spend.tolist(),
                        "planned": c.planned.tolist(),
                        "readout": c.readout,
                        "se": c.se,
                        "share": c.share,
                    }
                    for c in candidates
                ],
                "posterior": {
                    k: base_record[k]
                    for k in ("retention", "half_saturation", "coefficient", "chain", "length")
                },
                "sets": [
                    {
                        "tests": [
                            next(k for k, d in enumerate(candidates) if d is c)
                            for c in value.candidates
                        ],
                        "refitted": value.refitted.tolist(),
                        "reweight": value.reductions.tolist(),
                    }
                    for value in choice.values
                ],
                "refits": {
                    "base": {"divergent": base_record["divergent"], "rhat": base_record["rhat"]},
                    "divergent": [r["divergent"] for r in records],
                    "rhat": [r["rhat"] for r in records],
                },
            }
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
