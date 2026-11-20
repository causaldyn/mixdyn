"""Records Stan's reading of mixdyn's Bayesian media-mix model as mixdyn's oracle fixture
tests/fixtures/oracles/bayes_mmm_stan.json.

The model is written a second time, as a Stan program, its parameters declared in the order and on
the transforms of mixdyn's sampling coordinates, so that one point of those coordinates is one
point of both. Three worlds, each a national panel of 104 weeks with three channels, a promotion
held at or above 0, a price, a trend bent once and a yearly season of two harmonics, its KPI drawn
from the model at known parameters, three geo tests, and three lift estimates, a go-dark and a
heavy-up in sub-markets and a national heavy-up:

* ``normal``: Hill curves and normal noise;
* ``student``: Hill curves and Student-t noise of 5 degrees of freedom, fitted as Student-t;
* ``tanh``: ``tanh`` curves and normal noise;
* ``lognormal``: Hill curves and log-normal noise about the mean, the log's scale ``LOG_SIGMA``.

For each, CmdStan's ``log_prob`` gives the log density with its Jacobian and the gradient at 25
points, uniform in ``[-2, 2]`` on the sampling coordinates and moved into the support as mixdyn
moves its starts, and ``sample`` 4 chains of 10 000 draws after 1000 of warm-up, run in parallel
threads, summarised in the panel's units by mixdyn's own map from the sampling coordinates: each
parameter's posterior mean and standard deviation, with their Monte Carlo standard errors from batch
means, ``BATCHES`` batches a chain (the standard deviation's by the delta method from its square's),
and its R-hat; and each observation's log-likelihood, the KPI's observed weeks, then the tests'
readout periods, then the estimates, Stan's generated quantities summarised the same way and
recorded as they are at 25 of Stan's draws, spread evenly over the chains. The model's data are
mixdyn's, read off the panel by the module the fixture checks, but for the KPI, the spend, the tests
and the estimates, which are laid out here from the panel, the tests and the estimates themselves.
Each point is a ``log_prob`` run of its own: given several, CmdStan 2.38 reads them in the wrong
order, its reader taking the values in column-major order and reshaping them row by row. The
log-normal world's chains start where mixdyn's would, uniform and moved into the support: there a
uniform start's mean is seldom above 0 in every period, 599 of 200 000, so Stan's own starts,
uniform and drawn again at most 100 times, start a chain one time in four and all four chains about
one time in 220. The other worlds' chains start where Stan's own do.

CmdStan comes from the author's install, never a dependency:

    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu timeout 21600 uv run --project <mixdyn checkout> --no-sync \\
        python evidence/stan_bayes_mmm.py <cmdstan home> <work directory> <fixture path>
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import jax
import numpy as np
from chc.lift import GeoArm, LiftTest

from mixdyn import Cadence, Channel, Geo, MediaPanel, Revenue
from mixdyn.calibration import LiftEstimate
from mixdyn.inference.diagnostics import diagnose
from mixdyn.models.bayes_mmm import (
    BayesMedia,
    BayesSpec,
    HillCurve,
    Likelihood,
    LogNormal,
    Normal,
    Saturation,
    StudentT,
    TanhCurve,
    UniformStart,
    _initial,
    bayes_model,
)
from mixdyn.models.ridge_mmm import Control, FourierSeason, LinearTrend

jax.config.update("jax_enable_x64", True)

PROGRAM = """functions {
  vector adstock(vector x, real r) {
    int n = rows(x);
    vector[n] w;
    w[1] = 1;
    for (l in 2:n) {
      w[l] = w[l - 1] * r;
    }
    w /= sum(w);
    vector[n] a;
    for (t in 1:n) {
      a[t] = dot_product(w[1:t], reverse(x[1:t]));
    }
    return a;
  }
  vector hill(vector a, real k, real n) {
    vector[rows(a)] h;
    for (t in 1:rows(a)) {
      h[t] = a[t] > 0 ? inv_logit(n * (log(a[t]) - log(k))) : 0;
    }
    return h;
  }
  vector saturate(vector a, real k, vector slope, int m) {
    if (rows(slope) == 0) {
      return tanh(atanh(0.5) * a / k);
    }
    return hill(a, k, slope[m]);
  }
  matrix saturated(matrix spend, array[] int window, vector retention, vector half,
                   vector slope) {
    matrix[size(window), cols(spend)] h;
    for (m in 1:cols(spend)) {
      vector[rows(spend)] a = adstock(spend[:, m], retention[m]);
      h[:, m] = saturate(a[window], half[m], slope, m);
    }
    return h;
  }
  vector tested(matrix treated, matrix untreated, array[] int test_channel, array[] int readout,
                vector beta, vector retention, vector half, vector slope, int R) {
    vector[R] mean;
    int r = 1;
    for (k in 1:size(readout)) {
      int m = test_channel[k];
      int L = cols(treated);
      vector[L] ht = saturate(adstock(treated[k]', retention[m]), half[m], slope, m);
      vector[L] hc = saturate(adstock(untreated[k]', retention[m]), half[m], slope, m);
      int n = L - readout[k] + 1;
      mean[r:(r + n - 1)] = beta[m] * (ht[readout[k]:L] - hc[readout[k]:L]);
      r += n;
    }
    return mean;
  }
  vector estimated(matrix as_run, matrix as_planned, array[] int estimate_channel,
                   array[] int estimate_readout, vector beta, vector retention, vector half,
                   vector slope) {
    int L = cols(as_run);
    vector[size(estimate_readout)] mean;
    for (e in 1:size(estimate_readout)) {
      int m = estimate_channel[e];
      vector[L] hr = saturate(adstock(as_run[e]', retention[m]), half[m], slope, m);
      vector[L] hp = saturate(adstock(as_planned[e]', retention[m]), half[m], slope, m);
      mean[e] = beta[m] * sum(hr[estimate_readout[e]:L] - hp[estimate_readout[e]:L]);
    }
    return mean;
  }
}
data {
  int<lower=1> L;
  int<lower=1> N;
  array[N] int<lower=1, upper=L> window;
  int<lower=0, upper=N> O;
  array[O] int<lower=1, upper=N> observed;
  vector[O] y;
  int<lower=1> M;
  matrix[L, M] spend;
  vector<lower=0>[M] window_spend;
  vector[M] roi_mu;
  vector<lower=0>[M] roi_sigma;
  int<lower=0> P;
  matrix[N, P] calendar;
  int<lower=0> C;
  matrix[N, C] controls;
  vector[C] control_lower;
  vector[C] control_upper;
  int<lower=0, upper=1> student;
  int<lower=0, upper=1> tanh_curve;
  int<lower=0, upper=1> lognormal;
  int<lower=0> K;
  array[K] int<lower=1, upper=M> test_channel;
  matrix[K, L] treated;
  matrix[K, L] untreated;
  array[K] int<lower=1, upper=L> readout;
  int<lower=0> R;
  vector[R] gap;
  int<lower=0> E;
  array[E] int<lower=1, upper=M> estimate_channel;
  matrix[E, L] as_run;
  matrix[E, L] as_planned;
  array[E] int<lower=1, upper=L> estimate_readout;
  vector[E] lift;
  vector<lower=0>[E] lift_se;
}
parameters {
  vector[M] log_roi;
  vector<lower=0, upper=1>[M] retention;
  vector<lower=0>[M] half;
  vector<lower=0>[tanh_curve ? 0 : M] slope;
  real intercept;
  vector[P] calendar_coef;
  vector<lower=control_lower, upper=control_upper>[C] control_coef;
  real<lower=0> sigma;
  array[student] real<lower=0> nu;
  array[K > 0] real<lower=0> test_sigma;
}
model {
  matrix[N, M] h = saturated(spend, window, retention, half, slope);
  vector[M] beta = exp(log_roi) .* window_spend ./ (rep_row_vector(1, N) * h)';
  vector[N] mu = intercept + calendar * calendar_coef + controls * control_coef + h * beta;
  log_roi ~ normal(roi_mu, roi_sigma);
  retention ~ beta(1, 3);
  half ~ lognormal(0, 1);
  slope ~ lognormal(0, 0.5);
  intercept ~ normal(1, 1);
  calendar_coef ~ std_normal();
  control_coef ~ std_normal();
  sigma ~ std_normal();
  if (student) {
    nu[1] ~ gamma(2, 0.1);
    y ~ student_t(nu[1], mu[observed], sigma);
  } else if (lognormal) {
    y ~ lognormal(log(mu[observed]) - square(sigma) / 2, sigma);
  } else {
    y ~ normal(mu[observed], sigma);
  }
  if (K > 0) {
    test_sigma[1] ~ std_normal();
    gap ~ normal(tested(treated, untreated, test_channel, readout, beta, retention, half, slope,
                        R),
                 test_sigma[1]);
  }
  lift ~ normal(estimated(as_run, as_planned, estimate_channel, estimate_readout, beta, retention,
                          half, slope),
                lift_se);
}
generated quantities {
  vector[O + R + E] log_lik;
  {
    matrix[N, M] h = saturated(spend, window, retention, half, slope);
    vector[M] beta = exp(log_roi) .* window_spend ./ (rep_row_vector(1, N) * h)';
    vector[N] mu = intercept + calendar * calendar_coef + controls * control_coef + h * beta;
    for (o in 1:O) {
      if (student) {
        log_lik[o] = student_t_lpdf(y[o] | nu[1], mu[observed[o]], sigma);
      } else if (lognormal) {
        log_lik[o] = lognormal_lpdf(y[o] | log(mu[observed[o]]) - square(sigma) / 2, sigma);
      } else {
        log_lik[o] = normal_lpdf(y[o] | mu[observed[o]], sigma);
      }
    }
    if (K > 0) {
      vector[R] g = tested(treated, untreated, test_channel, readout, beta, retention, half,
                           slope, R);
      for (r in 1:R) {
        log_lik[O + r] = normal_lpdf(gap[r] | g[r], test_sigma[1]);
      }
    }
    vector[E] l = estimated(as_run, as_planned, estimate_channel, estimate_readout, beta,
                            retention, half, slope);
    for (e in 1:E) {
      log_lik[O + R + e] = normal_lpdf(lift[e] | l[e], lift_se[e]);
    }
  }
}
"""
WEEKS, SEED, POINTS, CHAINS, WARMUP, DRAWS, BATCHES = 104, 20261002, 25, 4, 1000, 10_000, 10
CHANNELS = ("tv", "search", "social")
TRUTH = {
    "roi": [1.5, 3.0, 0.8],
    "retention": [0.6, 0.1, 0.3],
    "half": [1.0, 0.7, 1.5],
    "slope": [1.2, 0.9, 2.0],
}
GEO = Geo("national")
WORLDS: dict[str, tuple[Likelihood, Saturation]] = {
    "normal": (Normal(), HillCurve()),
    "student": (StudentT(), HillCurve()),
    "tanh": (Normal(), TanhCurve()),
    "lognormal": (LogNormal(), HillCurve()),
}
LOG_SIGMA = 0.02
"""The log-normal world's noise on the log's scale, about the normal world's 20 about 1000."""


def saturated(
    spend: np.ndarray, retention: float, half: float, slope: float, saturation: Saturation
) -> np.ndarray:
    """A series carried over by the normalised kernel as long as it, then saturated."""
    w = retention ** np.arange(spend.size)
    a = np.convolve(spend, w / w.sum())[: spend.size]
    if isinstance(saturation, TanhCurve):
        return np.tanh(math.atanh(0.5) * a / half)
    return a**slope / (a**slope + half**slope)


def world(
    name: str,
) -> tuple[MediaPanel, dict[str, list[LiftTest]], dict[str, list[LiftEstimate]], BayesSpec]:
    likelihood, saturation = WORLDS[name]
    rng = np.random.default_rng([SEED, list(WORLDS).index(name)])
    spend = rng.gamma(2.0, 50.0, (WEEKS, 3)) * (rng.random((WEEKS, 3)) > 0.2)
    unit = np.array([s[s > 0].mean() for s in spend.T])
    t = np.arange(WEEKS)
    promotion = (rng.random(WEEKS) < 0.2).astype(float)
    price = 10.0 + rng.standard_normal(WEEKS)
    columns = []
    for j in range(3):
        shape = (TRUTH[k][j] for k in ("retention", "half", "slope"))
        h = saturated(spend[:, j] / unit[j], *shape, saturation)
        columns.append(TRUTH["roi"][j] * spend[:, j].sum() * h / h.sum())
    if isinstance(likelihood, StudentT):
        noise = rng.standard_t(5.0, WEEKS)
    else:
        noise = rng.standard_normal(WEEKS)
    mean = (
        1000.0
        + 0.8 * t
        + 50.0 * np.sin(2.0 * np.pi * t / 52.18)
        + 30.0 * promotion
        - 10.0 * (price - 10.0)
        + np.sum(columns, axis=0)
    )
    if isinstance(likelihood, LogNormal):
        kpi = mean * np.exp(LOG_SIGMA * noise - LOG_SIGMA**2 / 2.0)
    else:
        kpi = mean + 20.0 * noise
    panel = MediaPanel(
        dates=np.datetime64("2023-01-02") + np.timedelta64(7, "D") * t,
        cadence=Cadence.WEEK,
        geos=(GEO,),
        channels=tuple(Channel(c) for c in CHANNELS),
        controls=("promotion", "price"),
        kpi=kpi[None, :],
        kpi_unit=Revenue(),
        spend=spend[None],
        currency="EUR",
        control_values=np.column_stack([promotion, price])[None],
        filled=np.zeros((1, WEEKS), dtype=bool),
    )
    tests: dict[str, list[LiftTest]] = {}
    for j, factor in ((0, 0.0), (0, 2.0), (1, 0.0)):
        start = int(rng.integers(0, WEEKS - 12))
        control = spend[start : start + 12, j] * 0.1
        treated = np.concatenate([control[:8], control[8:] * factor])
        shape = [TRUTH[k][j] for k in ("retention", "half", "slope")]
        h = saturated(spend[:, j] / unit[j], *shape, saturation)
        beta = TRUTH["roi"][j] * spend[:, j].sum() / h.sum()
        gap = beta * np.subtract(
            *(
                saturated(np.r_[np.zeros(WEEKS - 12), s / 0.1 / unit[j]], *shape, saturation)[-4:]
                for s in (treated, control)
            )
        )
        base = rng.normal(100.0, 2.0, 4)
        tests.setdefault(CHANNELS[j], []).append(
            LiftTest(
                GeoArm(treated, 0.1 * (base / 0.1 + gap + rng.normal(0.0, 15.0, 4)), 0.1),
                GeoArm(control, base, 0.1),
            )
        )
    estimates: dict[str, list[LiftEstimate]] = {}
    for j, weeks, changed, factor, readout, share, se in (
        (2, 12, slice(8, 12), 0.0, 4, 0.2, 20.0),
        (2, 14, slice(8, 12), 2.0, 6, 1.0, 60.0),
        (0, 10, slice(6, 10), 1.5, 4, 0.1, 8.0),
    ):
        start = int(rng.integers(0, WEEKS - weeks))
        planned = spend[start : start + weeks, j] * share
        run = planned.copy()
        run[changed] *= factor
        shape = [TRUTH[k][j] for k in ("retention", "half", "slope")]
        h = saturated(spend[:, j] / unit[j], *shape, saturation)
        beta = TRUTH["roi"][j] * spend[:, j].sum() / h.sum()
        as_run, as_planned = (
            saturated(np.r_[np.zeros(WEEKS - weeks), s / share / unit[j]], *shape, saturation)
            for s in (run, planned)
        )
        lift = beta * (as_run - as_planned)[-readout:].sum()
        estimates.setdefault(CHANNELS[j], []).append(
            LiftEstimate(run, planned, readout, share * lift + rng.normal(0.0, se), se, share)
        )
    spec = BayesSpec(
        tuple(BayesMedia(c) for c in CHANNELS),
        (Control("promotion", lower=0.0), Control("price")),
        LinearTrend(1),
        FourierSeason(52.18, 2),
        likelihood,
        saturation,
    )
    return panel, tests, estimates, spec


def summary(x: np.ndarray) -> dict[str, list[float]]:
    """``x`` is ``(chains, draws, quantities)``: each quantity's mean and standard deviation, their
    standard errors from ``BATCHES`` batch means a chain, and its R-hat."""
    chains, draws, k = x.shape
    size = draws // BATCHES
    mean, sd = x.mean(axis=(0, 1)), x.std(axis=(0, 1))
    batches = x[:, : size * BATCHES].reshape(chains * BATCHES, size, k)
    n = math.sqrt(chains * BATCHES)
    squares = ((batches - mean) ** 2).mean(axis=1)
    return {
        "mean": mean.tolist(),
        "sd": sd.tolist(),
        "mean_se": (batches.mean(axis=1).std(axis=0, ddof=1) / n).tolist(),
        "sd_se": (squares.std(axis=0, ddof=1) / n / (2.0 * sd)).tolist(),
        "rhat": diagnose(x).rhat.tolist(),
    }


def data(
    model,
    panel: MediaPanel,
    tests: dict[str, list[LiftTest]],
    estimates: dict[str, list[LiftEstimate]],
) -> dict[str, object]:
    """The program's data on the model's units: the spend, its window, the tests and the estimates
    laid out from the panel, the tests and the estimates themselves, the rest read off the model."""
    names = [m.channel for m in model.spec.media]
    length, weeks = model.lagged.shape[2], panel.dates.size
    spend = panel.spend[0][:, [CHANNELS.index(c) for c in names]]
    arms, read = [], []
    for j, c in enumerate(names):
        for t in tests.get(c, []):
            groups = [
                np.r_[np.zeros(length - g.spend.size), g.spend / g.share / model.spend_unit[j]]
                for g in (t.treated, t.control)
            ]
            arms.append(
                (j, *groups, length - t.treated.outcome.size + 1, t.difference / model.kpi_unit)
            )
        for e in estimates.get(c, []):
            series = [
                np.r_[np.zeros(length - s.size), s / e.share / model.spend_unit[j]]
                for s in (e.spend, e.planned)
            ]
            read.append((j, *series, length - e.readout + 1, e.lift / e.share, e.se / e.share))
    return {
        "L": length,
        "N": weeks,
        "window": list(range(length - weeks + 1, length + 1)),
        "O": int(model.observed.sum()),
        "observed": (np.flatnonzero(model.observed) + 1).tolist(),
        "y": (panel.kpi[0][model.observed] / model.kpi_unit).tolist(),
        "M": len(names),
        "spend": np.r_[np.zeros((length - weeks, len(names))), spend / model.spend_unit].tolist(),
        "window_spend": (spend.sum(axis=0) / model.kpi_unit).tolist(),
        "roi_mu": [m.roi.mu for m in model.spec.media],
        "roi_sigma": [m.roi.sigma for m in model.spec.media],
        "P": model.calendar.shape[1],
        "calendar": model.calendar.tolist(),
        "C": model.controls.shape[1],
        "controls": model.controls.tolist(),
        "control_lower": [c.lower for c in model.spec.controls],
        "control_upper": [c.upper for c in model.spec.controls],
        "student": int(isinstance(model.spec.likelihood, StudentT)),
        "tanh_curve": int(isinstance(model.spec.saturation, TanhCurve)),
        "lognormal": int(isinstance(model.spec.likelihood, LogNormal)),
        "K": len(arms),
        "test_channel": [a[0] + 1 for a in arms],
        "treated": [a[1].tolist() for a in arms],
        "untreated": [a[2].tolist() for a in arms],
        "readout": [a[3] for a in arms],
        "R": sum(a[4].size for a in arms),
        "gap": [float(g) for a in arms for g in a[4]],
        "E": len(read),
        "estimate_channel": [r[0] + 1 for r in read],
        "as_run": [r[1].tolist() for r in read],
        "as_planned": [r[2].tolist() for r in read],
        "estimate_readout": [r[3] for r in read],
        "lift": [r[4] / model.kpi_unit for r in read],
        "lift_se": [r[5] / model.kpi_unit for r in read],
    }


def csv(path: Path) -> tuple[list[str], np.ndarray]:
    lines = [line for line in path.read_text().splitlines() if line and line[0] != "#"]
    return lines[0].split(","), np.array(
        [[float(v) for v in line.split(",")] for line in lines[1:]]
    )


def stan(executable: Path, work: Path, name: str, arguments: list[str], output: str) -> Path:
    path = work / f"{name}.{output}.csv"
    subprocess.run(
        [str(executable), *arguments, "data", f"file={work / f'{name}.data.json'}",
         "output", f"file={path}", "sig_figs=18", "refresh=0"],
        check=True,
        capture_output=True,
        timeout=7200,
    )  # fmt: skip
    return path


def case(executable: Path, work: Path, name: str) -> dict[str, object]:
    panel, tests, estimates, spec = world(name)
    model = bayes_model(panel, GEO, spec, tests=tests, estimates=estimates)
    (work / f"{name}.data.json").write_text(json.dumps(data(model, panel, tests, estimates)))
    points = np.array(
        [
            model._lift(x)
            for x in np.random.default_rng([SEED, list(WORLDS).index(name), 1]).uniform(
                -2.0, 2.0, (POINTS, model.dimension)
            )
        ]
    )
    rows = []
    for i, point in enumerate(points):
        source = work / f"{name}.point_{i}.json"
        source.write_text(json.dumps({"params_r": point.tolist()}))
        _, row = csv(
            stan(
                executable,
                work,
                name,
                ["log_prob", f"unconstrained_params={source}"],
                f"log_prob_{i}",
            )
        )
        rows.append(row[0])
    log_prob = np.array(rows)
    inits = []
    if isinstance(model.spec.likelihood, LogNormal):
        for c, start in enumerate(_initial(model, UniformStart(), CHAINS, SEED), start=1):
            inits.append(work / f"{name}.init_{c}.json")
            inits[-1].write_text(json.dumps(constrained(model, start)))
    stan(
        executable,
        work,
        name,
        ["sample", f"num_samples={DRAWS}", f"num_warmup={WARMUP}", f"num_chains={CHAINS}",
         *([f"init={','.join(map(str, inits))}"] if inits else []),
         "random", f"seed={SEED}", f"num_threads={CHAINS}"],
        "sample",
    )  # fmt: skip
    header, chains = None, []
    for c in range(1, CHAINS + 1):
        header, values = csv(work / f"{name}.sample_{c}.csv")
        chains.append(values)
    draws = np.array(chains)
    theta = unconstrained(header, draws.reshape(-1, draws.shape[2]), model)
    raw = model.parameters(theta)
    posterior = {key: summary(values.reshape(CHAINS, DRAWS, -1)) for key, values in raw.items()}
    divergent = int(draws[:, :, header.index("divergent__")].sum())
    log_lik = draws[:, :, [i for i, h in enumerate(header) if h.startswith("log_lik.")]]
    spread = np.linspace(0, CHAINS * DRAWS - 1, POINTS).round().astype(int)
    return {
        "panel": {
            "kpi": panel.kpi[0].tolist(),
            "spend": panel.spend[0].tolist(),
            "controls": panel.control_values[0].tolist(),
        },
        "tests": {
            c: [
                {
                    arm: {
                        "spend": getattr(t, arm).spend.tolist(),
                        "outcome": getattr(t, arm).outcome.tolist(),
                        "share": getattr(t, arm).share,
                    }
                    for arm in ("treated", "control")
                }
                for t in ts
            ]
            for c, ts in tests.items()
        },
        "estimates": {
            c: [
                {
                    "spend": e.spend.tolist(),
                    "planned": e.planned.tolist(),
                    "readout": e.readout,
                    "lift": e.lift,
                    "se": e.se,
                    "share": e.share,
                }
                for e in es
            ]
            for c, es in estimates.items()
        },
        "likelihood": type(spec.likelihood).__name__,
        "saturation": type(spec.saturation).__name__,
        "points": points.tolist(),
        "log_prob": log_prob[:, 0].tolist(),
        "gradient": log_prob[:, 1:].tolist(),
        "posterior": posterior,
        "log_lik": summary(log_lik),
        "log_lik_draws": {
            "theta": theta[spread].tolist(),
            "log_lik": log_lik.reshape(CHAINS * DRAWS, -1)[spread].tolist(),
        },
        "divergent": divergent,
    }


def constrained(model, theta: np.ndarray) -> dict[str, object]:
    """Stan's parameters at a point of mixdyn's sampling coordinates, for a chain's start."""
    p, _ = model._constrained(jax.numpy.asarray(theta))
    names = {
        "log_roi": "log_roi",
        "retention": "retention",
        "half": "half",
        "slope": "slope",
        "calendar": "calendar_coef",
        "control": "control_coef",
        "nu": "nu",
        "test_sigma": "test_sigma",
    }
    out: dict[str, object] = {stan: np.asarray(p[ours]).tolist() for ours, stan in names.items()}
    return {**out, "intercept": float(p["intercept"][0]), "sigma": float(p["sigma"][0])}


def unconstrained(header: list[str], draws: np.ndarray, model) -> np.ndarray:
    """Stan's draws on mixdyn's sampling coordinates, the inverse of each block's transform."""

    def block(prefix: str) -> np.ndarray:
        names = [h for h in header if h == prefix or h.startswith(prefix + ".")]
        return draws[:, [header.index(h) for h in names]]

    signs = np.array([c.lower != -np.inf or c.upper != np.inf for c in model.spec.controls])
    control = block("control_coef")
    columns = [
        block("log_roi"),
        np.log(block("retention") / (1.0 - block("retention"))),
        np.log(block("half")),
        np.log(block("slope")),
        block("intercept"),
        block("calendar_coef"),
        np.where(signs, np.log(np.abs(np.where(signs, control, 1.0))), control),
        np.log(block("sigma")),
        np.log(block("nu")),
        np.log(block("test_sigma")),
    ]
    theta = np.column_stack(columns)
    if theta.shape[1] != model.dimension:
        msg = f"Stan's draws give {theta.shape[1]} coordinates; the model has {model.dimension}"
        raise ValueError(msg)
    return theta


def main() -> None:
    cmdstan, work, fixture = (Path(a).resolve() for a in sys.argv[1:4])
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
    version = (cmdstan / "makefile").read_text().split("CMDSTAN_VERSION := ")[1].split()[0]
    fixture.write_text(
        json.dumps(
            {
                "provenance": {
                    "generator": "CmdStan's log_prob and sample on `program`, four national worlds",
                    "cmdstan": version,
                    "program": PROGRAM,
                    "seed": SEED,
                    "chains": CHAINS,
                    "warmup": WARMUP,
                    "draws": DRAWS,
                    "batches": BATCHES,
                    "truth": TRUTH,
                },
                "cases": {name: case(executable, work, name) for name in WORLDS},
            }
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
