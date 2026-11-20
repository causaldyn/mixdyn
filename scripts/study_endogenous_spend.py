"""Spend set by demand: the endogenous-spend flag, the sensitivity bound and the demand proxy on
the worlds that set it (ADR 0061, ADR 0074). Shares, intervals and returns only, no wall time.

Four studies, each writing one JSON line a job to `<out>/<study>.jsonl`, a job already written
skipped:

- `flag`: `mixdyn.data.feedback.spend_feedback` at its defaults on Track M v2's world
  (causaldyn-bench's `causaldyn_bench.endogenous_mmm`, Heusch 2026) at seeds 0 to `--seeds` - 1:
  `reference`, his reference instance; `exogenous`, its four mechanisms off; and each mechanism
  alone, `budget`, `anticipatory`, `bursts` and `bidding`; and on the search-demand world of
  `tests/search_demand_world.py`. A cell's share is that of seeds whose interval excludes 0.
- `sensitivity`: `mixdyn.inference.sensitivity.msm_sensitivity` of every channel, on the
  one-geo Bayesian model fitted to Track M v2's `reference` and `exogenous` worlds as the
  scorecard's arm reads an export, its tanh family: the promotion held at or above 0 and the
  price free, a trend of 3 bends, 3 harmonics of the year, Meridian's prior on each return; 2
  chains of 500 draws after 500 of warm-up from the world's seed, against the arm's 4 of 1000.
  And on the sensitivity tests' own exogenous world (`tests/test_sensitivity.py`, `flighted`),
  fitted as they fit it. Each job's draws of each return and each draw's allowance per unit of
  `kappa` are kept beside it, `<out>/sensitivity_<kind>_<seed>.npz`.
- `demand`: the search-demand world at seeds 0 to `--seeds` - 1, fitted by the one-geo Bayesian
  model, its tanh family, a season of one harmonic of 52 weeks, Meridian's prior on each return,
  with the queries a `DemandProxy` of search in each role: `confounder`, the queries a control,
  and `mediator`, the queries left out; and `nested`, the nested model (ADR 0014) whose stage one
  reads the queries on television, the default prior on its return moved to the queries' units by
  the ratio of their mean to the KPI's, and whose stage two is the confounder's model; and
  `nested, trend`, the same with a trend of 3 bends in stage one, a term of its own for the
  queries' drift; and `nested, half-normal` and `nested, half-normal, trend`, those two with
  stage one's prior `HalfNormalRoi()`'s moved to the queries' units by the same ratio (ADR 0076);
  2 chains of 500 after 500 each.
- `width`: the marginal sensitivity model's allowance at `Gamma = 1.5` over each channel's return
  on Track M v2's `reference` and `exogenous` worlds at seeds 0 to `--seeds` - 1, read at the
  world's true curves and coefficients, no posterior: each channel's true saturated spend `D`, its
  effect over its coefficient, and `X` the arm's other columns (the intercept, the promotion and
  price as read, the trend of 3 bends, 3 harmonics of the year) with the other channels' true
  saturated spend. `sigma` is the root mean square of the KPI less the true media after its least
  squares on `X`, `sd(D | X)` that of `D` after its least squares on `X` and the other channels,
  and the allowance over the return `kappa(1.5) sigma / (beta sd(D | X))`, which is below 0.1
  only where `beta sd(D | X) / sigma` passes `10 kappa(1.5) = 3.54`. Each part of the KPI the
  world draws and `X` does not hold, after the same least squares, is kept beside it, and the
  allowance with `sigma` the world's white noise alone, which no reading of its KPI goes below.

`--summary` reads the studies back as the tables ADR 0061's amendment and ADR 0074 state, each share
with its Monte Carlo standard error over the seeds, `sqrt(p (1 - p) / seeds)`. The runs those ADRs
state, from the repository's root, the bench on `PYTHONPATH`:

    PYTHONPATH=src:tests:<causaldyn-bench>/src python scripts/study_endogenous_spend.py \\
        --out out flag --seeds 1000
    PYTHONPATH=src:tests:<causaldyn-bench>/src python scripts/study_endogenous_spend.py \\
        --out out sensitivity --seeds 10
    PYTHONPATH=src:tests:<causaldyn-bench>/src python scripts/study_endogenous_spend.py \\
        --out out demand --seeds 10
    PYTHONPATH=src:tests:<causaldyn-bench>/src python scripts/study_endogenous_spend.py \\
        --out out width --seeds 10
    PYTHONPATH=src:tests:<causaldyn-bench>/src python scripts/study_endogenous_spend.py \\
        --out out --summary
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

os.environ.update({"JAX_ENABLE_X64": "1", "JAX_PLATFORMS": "cpu"})
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

SWITCHES = ("budget_feedback", "anticipatory", "bursts", "bidding")
FLAG_KINDS = {
    "reference": dict.fromkeys(SWITCHES, True),
    "exogenous": dict.fromkeys(SWITCHES, False),
    **{
        name: {s: s == switch for s in SWITCHES}
        for name, switch in zip(
            ("budget", "anticipatory", "bursts", "bidding"), SWITCHES, strict=True
        )
    },
}
GAMMAS = (1.0, 1.05, 1.1, 1.25, 1.5, 2.0, 3.0)
ROLES = (
    "confounder",
    "mediator",
    "nested",
    "nested, trend",
    "nested, half-normal",
    "nested, half-normal, trend",
)
CHAINS, WARMUP, DRAWS = 2, 500, 500


def _done(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _write(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _track_m2_panel(kind: str, seed: int) -> tuple[Any, tuple[str, ...], np.ndarray]:
    from causaldyn_bench.endogenous_mmm import EndogenousMediaMix

    from track_m2_worlds import TrackM2World

    w = EndogenousMediaMix(**FLAG_KINDS[kind]).simulate(seed)
    world = TrackM2World(
        kind,
        seed,
        w.channels,
        w.sales,
        w.spend,
        w.observed_promotion.astype(float),
        w.observed_price,
        w.roas(),
    )
    return world.panel(), w.channels, w.roas()


def flag(out: Path, seeds: int) -> None:
    from mixdyn import NATIONAL
    from mixdyn.data.feedback import spend_feedback
    from search_demand_world import world as search_world

    path = out / "flag.jsonl"
    done = {(r["kind"], r["seed"]) for r in _done(path)}
    for kind in (*FLAG_KINDS, "search-demand"):
        for seed in range(seeds):
            if (kind, seed) in done:
                continue
            panel = (
                search_world(seed).panel
                if kind == "search-demand"
                else _track_m2_panel(kind, seed)[0]
            )
            read = spend_feedback(panel, NATIONAL)
            _write(
                path,
                {
                    "kind": kind,
                    "seed": seed,
                    **{
                        f.channel: {"gamma": f.gamma, "se": f.se, "follows": f.follows}
                        for f in read
                    },
                },
            )


def sensitivity(out: Path, seeds: int) -> None:
    from mixdyn import NATIONAL
    from mixdyn.inference.sensitivity import msm_sensitivity
    from mixdyn.models.bayes_mmm import BayesMedia, BayesSpec, TanhCurve, fit_bayes
    from mixdyn.models.ridge_mmm import Control, FourierSeason, LinearTrend

    path = out / "sensitivity.jsonl"
    done = {(r["kind"], r["seed"]) for r in _done(path)}
    jobs = [(k, s) for s in range(seeds) for k in ("reference", "exogenous")]
    jobs += [("flighted", s) for s in range(seeds)]
    for kind, seed in jobs:
        if (kind, seed) in done:
            continue
        if kind == "flighted":
            from test_sensitivity import _confounded

            panel, roi, _ = _confounded(seed, 0.0, 0.0)
            channels, roas = ("tv",), np.array([roi])
            spec = BayesSpec((BayesMedia("tv"), BayesMedia("search")))
            fit = fit_bayes(panel, NATIONAL, spec, seed=seed, chains=2, warmup=300, draws=300)
        else:
            panel, channels, roas = _track_m2_panel(kind, seed)
            spec = BayesSpec(
                media=tuple(BayesMedia(c) for c in channels),
                controls=(Control("promotion", lower=0.0), Control("price")),
                trend=LinearTrend(3),
                season=FourierSeason(365.25 / 7.0, 3),
                saturation=TanhCurve(),
            )
            fit = fit_bayes(
                panel, NATIONAL, spec, seed=seed, chains=CHAINS, warmup=WARMUP, draws=DRAWS
            )
        rhat = np.concatenate([d.rhat for d in fit.diagnostics.values()])
        record: dict[str, Any] = {
            "kind": kind,
            "seed": seed,
            "channels": list(channels),
            "roas": roas.tolist(),
            "divergent": fit.divergent.tolist(),
            "rhat_max": float(np.nanmax(rhat)),
            "gammas": list(GAMMAS),
        }
        arrays = {}
        names = [m.channel for m in fit.spec.media]
        for c in channels:
            m = msm_sensitivity(fit, c, GAMMAS)
            roi = fit.draws["roi"][..., names.index(c)].reshape(-1)
            record[c] = {
                "roi": m.roi,
                "gamma_star": m.gamma_star,
                "lower": m.lower.tolist(),
                "upper": m.upper.tolist(),
                "allowance": m.allowance.tolist(),
            }
            arrays[f"{c}_roi"], arrays[f"{c}_scale"] = roi, m.scale
        np.savez_compressed(out / f"sensitivity_{kind}_{seed}.npz", **arrays)
        _write(path, record)


def demand(out: Path, seeds: int) -> None:
    from mixdyn import NATIONAL
    from mixdyn.models.bayes_mmm import (
        BayesMedia,
        BayesSpec,
        HalfNormalRoi,
        RoiPrior,
        TanhCurve,
        fit_bayes,
    )
    from mixdyn.models.demand import DemandProxy
    from mixdyn.models.nested import NestedSpec, fit_nested
    from mixdyn.models.ridge_mmm import FourierSeason, LinearTrend
    from search_demand_world import CHANNELS
    from search_demand_world import world as search_world

    path = out / "demand.jsonl"
    done = {(r["role"], r["seed"]) for r in _done(path)}
    season, curve = FourierSeason(52.0, 1), TanhCurve()
    for seed in range(seeds):
        w = search_world(seed)
        for role in ROLES:
            if (role, seed) in done:
                continue
            nested = role not in ("confounder", "mediator")
            confounder = DemandProxy("queries", "confounder", ("search",))
            proxy = (
                DemandProxy("queries", role, ("search",))
                if role in ("confounder", "mediator")
                else confounder
            )
            spec = BayesSpec(
                media=tuple(BayesMedia(c) for c in CHANNELS),
                controls=proxy.controls(),
                season=season,
                saturation=curve,
            )
            if nested:
                queries = w.panel.control_values[0, :, w.panel.controls.index("queries")]
                ratio = queries.mean() / w.panel.kpi[0].mean()
                prior = (
                    HalfNormalRoi(HalfNormalRoi().scale * ratio)
                    if "half-normal" in role
                    else RoiPrior(RoiPrior().mu + math.log(ratio), RoiPrior().sigma)
                )
                stage_one = BayesSpec(
                    media=(BayesMedia("tv", prior),),
                    trend=LinearTrend(3) if role.endswith("trend") else None,
                    season=season,
                    saturation=curve,
                )
                fit = fit_nested(
                    w.panel,
                    NATIONAL,
                    NestedSpec("queries", stage_one, spec),
                    seed=seed,
                    chains=CHAINS,
                    warmup=WARMUP,
                    draws=DRAWS,
                )
                returns = fit.roi()
                roi = np.column_stack([returns[c].reshape(-1) for c in CHANNELS])
                divergent = [fit.stage_one.divergent.tolist(), fit.stage_two.divergent.tolist()]
                rhat = max(float(np.nanmax(d.rhat)) for d in fit.stage_one.diagnostics.values())
            else:
                one = fit_bayes(
                    w.panel, NATIONAL, spec, seed=seed, chains=CHAINS, warmup=WARMUP, draws=DRAWS
                )
                roi = one.draws["roi"].reshape(-1, len(CHANNELS))
                divergent = one.divergent.tolist()
                rhat = max(float(np.nanmax(d.rhat)) for d in one.diagnostics.values())
            _write(
                path,
                {
                    "role": role,
                    "seed": seed,
                    "truth": {"search": w.search, "tv direct": w.tv_direct, "tv total": w.tv_total},
                    "divergent": divergent,
                    "rhat": rhat,
                    **{
                        c: [float(np.quantile(roi[:, j], q)) for q in (0.05, 0.5, 0.95)]
                        for j, c in enumerate(CHANNELS)
                    },
                },
            )


def _residual(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    return y - x @ np.linalg.lstsq(x, y, rcond=None)[0]


def _rms(v: np.ndarray) -> float:
    return math.sqrt(float(v @ v) / v.size)


def width(out: Path, seeds: int) -> None:
    from causaldyn_bench.endogenous_mmm import ELASTICITY, EndogenousMediaMix
    from chc import ChannelSensitivity

    from mixdyn.models.ridge_mmm import FourierSeason, LinearTrend

    path = out / "width.jsonl"
    done = {(r["kind"], r["seed"]) for r in _done(path)}
    trend, season = LinearTrend(3), FourierSeason(365.25 / 7.0, 3)
    kappa = ChannelSensitivity.factor(1.5)
    for kind in ("reference", "exogenous"):
        generator = EndogenousMediaMix(**FLAG_KINDS[kind])
        for seed in range(seeds):
            if (kind, seed) in done:
                continue
            w = generator.simulate(seed)
            n = w.sales.size
            x = np.column_stack(
                [
                    np.ones(n),
                    w.observed_promotion.astype(float),
                    w.observed_price,
                    trend.columns(n) / n,
                    season.columns(n),
                ]
            )
            beta = np.array(w.effect)
            saturated = w.media / beta
            sigma = _rms(_residual(w.sales - w.media.sum(axis=1), x))
            structural = (
                generator.base
                * (1.0 + w.seasonality)
                * (1.0 + w.quality)
                * w.price_position**ELASTICITY
            )
            sentiment = generator.base * w.sentiment
            parts = {
                "promotion": w.promotion,
                "season, quality and price": structural,
                "sentiment": sentiment,
                "noise": w.baseline - structural - sentiment,
            }
            everything = np.column_stack([x, saturated])
            rms = {k: _rms(_residual(v, everything)) for k, v in parts.items()}
            record: dict[str, Any] = {"kind": kind, "seed": seed, "sigma": sigma, "parts": rms}
            for j, c in enumerate(w.channels):
                others = np.delete(saturated, j, axis=1)
                spread = _rms(_residual(saturated[:, j], np.column_stack([x, others])))
                signal = float(beta[j]) * spread
                record[c] = {
                    "roas": float(w.roas()[j]),
                    "log_spend_sd": _rms(_residual(np.log(w.spend[:, j]), x)),
                    "signal": signal,
                    "ratio": kappa * sigma / signal,
                    "floor": kappa * rms["noise"] / signal,
                }
            _write(path, record)


def _holding(roi: np.ndarray, scale: np.ndarray, truth: float) -> float:
    """The least `Gamma` whose bound, 5 % to 95 % of the draws' return less and plus their
    allowance, holds `truth`: each end moves one way along `kappa`, so each is bisected."""
    from chc import ChannelSensitivity

    def least(reached: Any) -> float:
        if reached(0.0):
            return 0.0
        below, above = 0.0, 1.0
        while not reached(above):
            below, above = above, 2.0 * above
        for _ in range(200):
            middle = 0.5 * (below + above)
            if middle in (below, above):
                break
            below, above = (below, middle) if reached(middle) else (middle, above)
        return above

    kappa = max(
        least(lambda k: float(np.quantile(roi - k * scale, 0.05)) <= truth),
        least(lambda k: float(np.quantile(roi + k * scale, 0.95)) >= truth),
    )
    return 1.0 if kappa == 0.0 else ChannelSensitivity.level_at(kappa)


def _spread(values: list[float]) -> str:
    v = np.array(values)
    return f"{np.median(v):.4g} [{v.min():.4g}, {v.max():.4g}]"


def _share(hits: list[bool]) -> str:
    p = float(np.mean(hits))
    return f"{p:.3f} ({math.sqrt(p * (1.0 - p) / len(hits)):.3f})"


def summary(out: Path) -> None:
    from chc import ChannelSensitivity

    rows = _done(out / "flag.jsonl")
    if rows:
        print("\n## flag: share of seeds whose interval excludes 0 (MC SE)\n")
        print("| world | seeds | " + " | ".join(("pla", "meta", "tv", "search")) + " |")
        print("|---|---|---|---|---|---|")
        for kind in (*FLAG_KINDS, "search-demand"):
            mine = [r for r in rows if r["kind"] == kind]
            if not mine:
                continue
            cells = [
                _share([r[c]["follows"] for r in mine]) if c in mine[0] else "-"
                for c in ("pla", "meta", "tv", "search")
            ]
            print(f"| {kind} | {len(mine)} | " + " | ".join(cells) + " |")
    rows = _done(out / "sensitivity.jsonl")
    if rows:
        print("\n## sensitivity: per channel over the seeds\n")
        print(
            "| world | channel | seeds | true ROAS | median roi | Gamma* median [min, max] |"
            " held at Gamma* | held at 1 | least Gamma that holds it, median [min, max] |"
            " allowance(1.5) / ROAS median [min, max] |"
        )
        print("|---|---|---|---|---|---|---|---|---|---|")
        for kind in ("reference", "exogenous", "flighted"):
            mine = [r for r in rows if r["kind"] == kind]
            if not mine:
                continue
            k15 = GAMMAS.index(1.5)
            for j, c in enumerate(mine[0]["channels"]):
                truth = np.array([r["roas"][j] for r in mine])
                star = np.array([r[c]["gamma_star"] for r in mine])
                held_star, held_one, cover = [], [], []
                for r, roas in zip(mine, truth, strict=True):
                    arrays = np.load(out / f"sensitivity_{kind}_{r['seed']}.npz")
                    roi, scale = arrays[f"{c}_roi"], arrays[f"{c}_scale"]
                    kappa = ChannelSensitivity.factor(r[c]["gamma_star"])
                    upper = float(np.quantile(roi + kappa * scale, 0.95))
                    held_star.append(bool(roas <= upper))
                    held_one.append(bool(r[c]["lower"][0] <= roas <= r[c]["upper"][0]))
                    cover.append(_holding(roi, scale, float(roas)))
                ratio = np.array([r[c]["allowance"][k15] for r in mine]) / truth
                held_at = np.array(cover)
                print(
                    f"| {kind} | {c} | {len(mine)} | {np.mean(truth):.3f} |"
                    f" {np.median([r[c]['roi'] for r in mine]):.3f} |"
                    f" {np.median(star):.4f} [{star.min():.4f}, {star.max():.4f}] |"
                    f" {_share(held_star)} | {_share(held_one)} |"
                    f" {np.median(held_at):.4f} [{held_at.min():.4f}, {held_at.max():.4f}] |"
                    f" {np.median(ratio):.3f} [{ratio.min():.3f}, {ratio.max():.3f}] |"
                )
    rows = _done(out / "width.jsonl")
    if rows:
        print("\n## width: at the true curves, median [min, max] over the seeds\n")
        print(
            "| world | seeds | sigma | promotion | season, quality and price | sentiment | noise |"
        )
        print("|---|---|---|---|---|---|---|")
        for kind in ("reference", "exogenous"):
            mine = [r for r in rows if r["kind"] == kind]
            if mine:
                cells = [_spread([r["sigma"] for r in mine])]
                cells += [_spread([r["parts"][k] for r in mine]) for k in mine[0]["parts"]]
                print(f"| {kind} | {len(mine)} | " + " | ".join(cells) + " |")
        print(
            "\n| world | channel | sd of log spend given X | beta sd(D given X) |"
            " allowance(1.5) / return | with sigma the noise alone |"
        )
        print("|---|---|---|---|---|---|")
        for kind in ("reference", "exogenous"):
            mine = [r for r in rows if r["kind"] == kind]
            for c in ("pla", "meta", "tv") if mine else ():
                cells = [
                    _spread([r[c][k] for r in mine])
                    for k in ("log_spend_sd", "signal", "ratio", "floor")
                ]
                print(f"| {kind} | {c} | " + " | ".join(cells) + " |")
    rows = _done(out / "demand.jsonl")
    if rows:
        print("\n## demand: returns' posterior median and 90 % interval held, per role\n")
        print(
            "| role | seeds | search median / truth | search held | tv median / direct |"
            " tv held direct | tv median / total | tv held total |"
        )
        print("|---|---|---|---|---|---|---|---|")
        for role in ("mediator", "confounder", *ROLES[2:]):
            mine = [r for r in rows if r["role"] == role]
            if not mine:
                continue

            def ratio(c: str, truth: str, mine: list[dict[str, Any]] = mine) -> str:
                values = np.array([r[c][1] / r["truth"][truth] for r in mine])
                return f"{np.median(values):.2f} [{values.min():.2f}, {values.max():.2f}]"

            def held(c: str, truth: str, mine: list[dict[str, Any]] = mine) -> str:
                return _share([r[c][0] <= r["truth"][truth] <= r[c][2] for r in mine])

            print(
                f"| {role} | {len(mine)} | {ratio('search', 'search')} | {held('search', 'search')}"
                f" | {ratio('tv', 'tv direct')} | {held('tv', 'tv direct')}"
                f" | {ratio('tv', 'tv total')} | {held('tv', 'tv total')} |"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("study", nargs="?", choices=("flag", "sensitivity", "demand", "width"))
    parser.add_argument("--seeds", type=int, default=10)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.study:
        {"flag": flag, "sensitivity": sensitivity, "demand": demand, "width": width}[args.study](
            args.out, args.seeds
        )
    if args.summary:
        summary(args.out)


if __name__ == "__main__":
    main()
