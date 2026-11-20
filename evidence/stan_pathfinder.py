"""Compares mixdyn's Pathfinder with Stan's on normal targets, path by path in distribution.

The two follow one algorithm (Zhang, Carpenter, Gelman and Vehtari 2022) through different
optimisers: mixdyn's path is SciPy's L-BFGS-B, Stan's its own L-BFGS with another line search and
first step, so no path is the other's and the comparison is of what the paths reach. Each target
is a normal with a known covariance; each of ``RUNS`` single paths starts uniform in ``[-2, 2]``,
as Stan's do, and draws ``DRAWS`` points from its best approximation. A run is scored by the
Kullback-Leibler divergence from the target of the normal fitted to its draws, and by its draws'
own estimate of that divergence, the mean of the approximation's log density less the target's.
The targets are a 5-dimensional normal, the unit tests', and a 20-dimensional one with condition
number 1000, wider than a path's history of 5 corrections.

CmdStan comes from the author's install, never a dependency:

    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 timeout 3600 \\
        uv run --project <mixdyn checkout> --no-sync \\
        python evidence/stan_pathfinder.py <cmdstan home> <work directory> <result path>
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from sbc_bayes_mmm import checkout

from mixdyn.inference.pathfinder import pathfinder

RUNS = 100
DRAWS = 1000
SEED = 20261002

PROGRAM = """data {
  int<lower=1> d;
  vector[d] mu;
  cov_matrix[d] sigma;
}
parameters {
  vector[d] q;
}
model {
  target += multi_normal_lpdf(q | mu, sigma);
}
"""


def targets() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(5)
    a = rng.normal(size=(5, 5))
    small = (np.array([1.0, -2.0, 0.5, 3.0, -1.0]), a @ a.T / 5 + 0.2 * np.eye(5))
    rng = np.random.default_rng(SEED)
    basis = np.linalg.qr(rng.normal(size=(20, 20)))[0]
    scales = np.logspace(0.0, 3.0, 20)
    wide = (rng.normal(size=20), basis @ np.diag(scales) @ basis.T / math.sqrt(1000.0))
    return {"normal5": small, "normal20": wide}


def divergence(draws: np.ndarray, mean: np.ndarray, covariance: np.ndarray) -> float:
    """KL(N(draws' mean, draws' covariance) || N(mean, covariance))."""
    m, s = draws.mean(axis=0), np.cov(draws.T)
    precision = np.linalg.inv(covariance)
    r = m - mean
    return 0.5 * float(
        np.trace(precision @ s)
        - mean.size
        + r @ precision @ r
        + np.linalg.slogdet(covariance)[1]
        - np.linalg.slogdet(s)[1]
    )


def stan_runs(executable: Path, work: Path, name: str, mean, covariance) -> list[dict]:
    data = work / f"{name}.json"
    data.write_text(json.dumps({"d": mean.size, "mu": mean.tolist(), "sigma": covariance.tolist()}))
    runs = []
    for run in range(RUNS):
        output = work / f"{name}_{run}.csv"
        subprocess.run(
            [
                str(executable),
                "pathfinder",
                "num_paths=1",
                f"num_draws={DRAWS}",
                "data",
                f"file={data}",
                "random",
                f"seed={SEED + run}",
                "output",
                f"file={output}",
            ],
            check=True,
            capture_output=True,
            timeout=120,
        )
        rows = [r for r in output.read_text().splitlines() if r and not r.startswith("#")]
        header = rows[0].split(",")
        table = np.array([[float(v) for v in r.split(",")] for r in rows[1:]])
        draws = table[:, [header.index(f"q.{i + 1}") for i in range(mean.size)]]
        gap = table[:, header.index("lp_approx__")] - table[:, header.index("lp__")]
        runs.append({"kl": divergence(draws, mean, covariance), "estimate": float(gap.mean())})
    return runs


def mixdyn_runs(mean, covariance) -> list[dict]:
    precision = jnp.asarray(np.linalg.inv(covariance))
    constant = -0.5 * (mean.size * math.log(2.0 * math.pi) + np.linalg.slogdet(covariance)[1])

    def log_density(q):
        r = q - mean
        return constant - 0.5 * r @ precision @ r

    runs = []
    for run in range(RUNS):
        start = np.random.default_rng(SEED + run).uniform(-2.0, 2.0, (1, mean.size))
        path = pathfinder(log_density, start, seed=SEED + run, draws=DRAWS)[0]
        runs.append(
            {
                "kl": divergence(path.draws, mean, covariance),
                "estimate": float(np.mean(path.log_q - path.log_p)),
            }
        )
    return runs


def summary(runs: list[dict]) -> dict:
    kl = np.array([r["kl"] for r in runs])
    gap = np.array([r["kl"] - r["estimate"] for r in runs])
    return {
        "kl_quantiles": dict(
            zip(
                ("10", "25", "50", "75", "90", "max"),
                np.quantile(kl, [0.1, 0.25, 0.5, 0.75, 0.9, 1.0]).round(4).tolist(),
                strict=True,
            )
        ),
        "estimate_less_fitted_median": round(float(np.median(gap)), 4),
    }


def main() -> None:
    if not jax.config.read("jax_enable_x64"):
        msg = "run with JAX_ENABLE_X64=1"
        raise SystemExit(msg)
    cmdstan, work, result = (Path(a).resolve() for a in sys.argv[1:4])
    work.mkdir(parents=True, exist_ok=True)
    source = work / "normal.stan"
    if not source.exists() or source.read_text() != PROGRAM:
        source.write_text(PROGRAM)
    executable = work / "normal"
    subprocess.run(
        ["make", "-C", str(cmdstan), str(executable)], check=True, capture_output=True, timeout=1200
    )
    version = (cmdstan / "makefile").read_text().split("CMDSTAN_VERSION := ")[1].split()[0]
    record: dict = {
        "mixdyn": checkout(),
        "cmdstan": version,
        "runs": RUNS,
        "draws": DRAWS,
        "seed": SEED,
        "targets": {},
    }
    for name, (mean, covariance) in targets().items():
        stan = stan_runs(executable, work, name, mean, covariance)
        ours = mixdyn_runs(mean, covariance)
        record["targets"][name] = {
            "stan": summary(stan),
            "mixdyn": summary(ours),
            "stan_runs": stan,
            "mixdyn_runs": ours,
        }
        print(name, "stan", summary(stan), "\n", name, "mixdyn", summary(ours), flush=True)
    result.write_text(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
