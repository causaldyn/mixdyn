"""Which prior should the engine's DLM start from? Counts and coverage only, no wall time.

Each world is 156 weekly steps of a random-walk level and two random-walk media coefficients
(step standard deviations 0.05 and 0.02), media drawn gamma(2, 1) and zero a quarter of the time,
and noise of variance 0.25. The level starts at 10, or at 100 in the `high` world, where the KPI's
mean is about fifty of its standard deviations. Every candidate fits the engine's model (a level
and the two effects, additive discounts) and differs in its prior or in the idle hold:

    ols         OLS on every step before the held-out tail gives m0 and C0, and V, fixed at the
                fit's residual variance. Those steps then enter twice.
    engine      `vague_prior`, the engine's default, with the default idle hold.
    no_hold     `vague_prior`, every coefficient discounted at every step.
    level_zero  `vague_prior` with the level's mean at zero instead of the KPI's mean.
    segment     OLS on the first 26 steps only, as a prior at step 26 with V learned from its
                residual degrees of freedom; the filter starts after the segment.

Measured per replicate, with the last 13 steps held out of every prior: the sum of one-step log
predictive densities over those 13 steps; the 90 % one-step coverage over steps 27-52 (early) and
over the last 13 (late); and the 90 % coverage of each channel's contribution summed over steps
27-39 (early) and over the last 13 (late), from 1000 backward draws. A difference of log scores
is reported with its standard error over replicates; a coverage's Monte Carlo standard error is
sqrt(0.9 * 0.1 / count).

    OMP_NUM_THREADS=1 uv run python scripts/study_dlm_prior.py [--replicates 500] > out.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from typing import Any

import numpy as np
from chc.dlm import DLMFit, Prior, backward_sample
from scipy import stats

from mixdyn import NATIONAL, Cadence, Count, MediaPanel, read_panel
from mixdyn.models import DlmSpec, Effect, Trend, fit_dlm, vague_prior

SEED = 20261001
LEVEL = 0.90
HORIZON = 156
HELDOUT = 13
SEGMENT = 26
EARLY = 26
DRAWS = 1000
DISCOUNTS = ((0.95, 0.95), (0.98, 0.99))
WORLDS = {"walk": 10.0, "high": 100.0}
CANDIDATES = ("ols", "engine", "no_hold", "level_zero", "segment")
CHANNELS = ("spend_a", "spend_b")


def world(rng: np.random.Generator, start: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    level = start + np.cumsum(0.05 * rng.standard_normal(HORIZON))
    coef = 1.0 + np.cumsum(0.02 * rng.standard_normal((HORIZON, 2)), axis=0)
    x = rng.gamma(2.0, 1.0, (HORIZON, 2)) * (rng.random((HORIZON, 2)) > 0.25)
    y = level + np.sum(x * coef, axis=1) + 0.5 * rng.standard_normal(HORIZON)
    return y, x, coef


def as_panel(y: np.ndarray, x: np.ndarray, first: int) -> MediaPanel:
    dates = np.datetime64("2023-01-02") + 7 * np.arange(first, first + y.size)
    data: dict[str, Any] = {"date": dates, "kpi": y}
    data |= {name: x[:, j] for j, name in enumerate(CHANNELS)}
    return read_panel(
        data,
        date="date",
        kpi="kpi",
        spend={name: name for name in CHANNELS},
        cadence=Cadence.WEEK,
        kpi_unit=Count("units"),
        currency="USD",
    )


def ols(y: np.ndarray, x: np.ndarray, dof_known: bool) -> Prior:
    design = np.column_stack([np.ones(y.size), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    residuals = y - design @ beta
    dof = y.size - design.shape[1]
    s2 = float(residuals @ residuals / dof)
    covariance = s2 * np.linalg.inv(design.T @ design)
    return Prior(beta, covariance, s2, math.inf if dof_known else float(dof))


def spec_of(discounts: tuple[float, float], hold: bool) -> DlmSpec:
    effects = tuple(Effect(name, discounts[1], hold_when_idle=hold) for name in CHANNELS)
    return DlmSpec(effects, trend=Trend(1, discounts[0]))


def fit(
    name: str, y: np.ndarray, x: np.ndarray, discounts: tuple[float, float]
) -> tuple[DLMFit, int]:
    """The candidate's filter, and the calendar step it starts at."""
    whole = as_panel(y, x, 0)
    train = HORIZON - HELDOUT
    if name == "ols":
        unheld = spec_of(discounts, hold=False)
        return fit_dlm(whole, NATIONAL, unheld, ols(y[:train], x[:train], True)).filtered, 0
    if name == "segment":
        rest = as_panel(y[SEGMENT:], x[SEGMENT:], SEGMENT)
        prior = ols(y[:SEGMENT], x[:SEGMENT], False)
        return fit_dlm(rest, NATIONAL, spec_of(discounts, True), prior).filtered, SEGMENT
    spec = spec_of(discounts, hold=name != "no_hold")
    prior = vague_prior(whole, NATIONAL, spec)
    if name == "level_zero":
        prior = dataclasses.replace(prior, mean=np.r_[0.0, prior.mean[1:]])
    return fit_dlm(whole, NATIONAL, spec, prior).filtered, 0


def covered(fit_: DLMFit, steps: slice) -> np.ndarray:
    u = fit_.errors[steps] / np.sqrt(fit_.one_step_scale[steps])
    pit = stats.t.cdf(u, fit_.one_step_dof[steps])
    return np.abs(pit - 0.5) <= LEVEL / 2.0


def contribution_hit(
    states: np.ndarray, x: np.ndarray, coef: np.ndarray, steps: slice, start: int
) -> np.ndarray:
    shifted = slice(steps.start - start, steps.stop - start)
    sampled = np.einsum("tj,dtj->dj", x[steps], states[:, shifted, 1:])
    truth = np.sum(x[steps] * coef[steps], axis=0)
    lo, hi = np.quantile(sampled, [0.5 - LEVEL / 2.0, 0.5 + LEVEL / 2.0], axis=0)
    return (lo <= truth) & (truth <= hi)


def run(replicates: int, start: float, discounts: tuple[float, float], seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    late = slice(HORIZON - HELDOUT, HORIZON)
    windows = {"early": slice(SEGMENT, SEGMENT + EARLY), "late": late}
    contribution_windows = {"early": slice(SEGMENT, SEGMENT + HELDOUT), "late": late}
    scores: dict[str, list[float]] = {c: [] for c in CANDIDATES}
    one_step: dict[str, dict[str, list[np.ndarray]]] = {
        c: {w: [] for w in windows} for c in CANDIDATES
    }
    hits = {c: {w: np.zeros(2) for w in windows} for c in CANDIDATES}
    for _ in range(replicates):
        y, x, coef = world(rng, start)
        draws_seed = int(rng.integers(2**31))
        for name in CANDIDATES:
            filtered, first = fit(name, y, x, discounts)
            scores[name].append(float(np.sum(filtered.log_scores[late.start - first :])))
            for w, steps in windows.items():
                local = slice(steps.start - first, steps.stop - first)
                one_step[name][w].append(covered(filtered, local))
            states = backward_sample(filtered, DRAWS, seed=draws_seed).states
            for w, steps in contribution_windows.items():
                hits[name][w] += contribution_hit(states, x, coef, steps, first)
    base = np.array(scores["ols"])
    out: dict[str, Any] = {"level": start, "discounts": discounts}
    for name in CANDIDATES:
        difference = np.array(scores[name]) - base
        out[name] = {
            "heldout_logscore": round(float(np.mean(scores[name])), 3),
            "minus_ols": [
                round(float(difference.mean()), 3),
                round(float(difference.std(ddof=1) / math.sqrt(replicates)), 3),
            ],
            **{
                f"one_step_{w}": round(float(np.concatenate(one_step[name][w]).mean()), 3)
                for w in windows
            },
            **{
                f"contribution_{w}": [round(float(v), 3) for v in hits[name][w] / replicates]
                for w in windows
            },
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=500)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    runs = [
        run(args.replicates, start, discounts, args.seed)
        for start in WORLDS.values()
        for discounts in DISCOUNTS
    ]
    print(json.dumps({"replicates": args.replicates, "seed": args.seed, "runs": runs}, indent=1))


if __name__ == "__main__":
    main()
