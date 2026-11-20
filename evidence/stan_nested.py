"""Records Stan's reading of mixdyn's nested media-mix model, its two stages drawn as one model
under lifts on the KPI of a channel that moves the mediator, as mixdyn's oracle fixture
tests/fixtures/oracles/nested_stan.json.

The model is written a second time, as one Stan program: each stage's block is the one-geo
program's of ``stan_bayes_mmm.py``, its data and parameters suffixed by the stage, stage one's
parameters declared first, each in the order and on the transforms of mixdyn's sampling
coordinates, so that one point of those coordinates is one point of both. Each lift on the KPI
through the mediator is normal about ``gamma`` times its lift on the mediator, plus its lift on
the KPI directly where stage two has the channel, ``gamma`` the mediator's coefficient in stage
two, standardised as stage two reads it, over the mediator's standard deviation and times stage
one's unit. Two worlds, each a national panel of 104 weeks where television moves branded searches
and searches and search ads move sales, sales falling with price:

* ``mediated``: television in stage one alone, both stages' curves Hill's; its lifts on sales a
  go-dark in a fifth of the market and a heavy-up in a quarter whose spend is recorded from ten
  weeks before the panel, so that stage one's kernels run past the calendar;
* ``both``: television in both stages, stage one's curve ``tanh``, stage two's Hill's; its lifts
  on sales a go-dark in a fifth of the market and a national heavy-up.

Each world has a lift on the searches of television and one on sales of the search ads besides,
which its stages read alone. For each, CmdStan's ``log_prob`` gives the log density with its
Jacobian and the gradient at 25 points uniform in ``[-2, 2]`` on the sampling coordinates, each a
run of its own, and ``sample`` 4 chains of 10 000 draws after 1000 of warm-up, run in parallel
threads, summarised in the panel's units by mixdyn's own maps from each stage's coordinates: each
parameter's posterior mean and standard deviation, with their Monte Carlo standard errors from
batch means, and its R-hat; and each observation's log-likelihood, stage one's searches and its
lift, stage two's sales and its lift, then the lifts through the mediator, on the model's scales,
summarised the same way and recorded as they are at 25 of Stan's draws spread evenly over the
chains. The data are mixdyn's, read off its model, but for the KPIs, the spend and the lifts,
which are laid out here from the panel and the estimates themselves.

CmdStan comes from the author's install, never a dependency:

    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu timeout 21600 uv run --project <mixdyn checkout> --no-sync \\
        python evidence/stan_nested.py <cmdstan home> <work directory> <fixture path>
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import jax
import numpy as np
from stan_bayes_mmm import PROGRAM as ONE_GEO
from stan_bayes_mmm import csv, stan, summary

from mixdyn import Cadence, Channel, Geo, MediaPanel, Revenue
from mixdyn.calibration import LiftEstimate
from mixdyn.models.bayes_mmm import BayesMedia, BayesSpec, HillCurve, RoiPrior, TanhCurve
from mixdyn.models.nested import NestedSpec, nested_model
from mixdyn.models.ridge_mmm import Control, FourierSeason

jax.config.update("jax_enable_x64", True)

FUNCTIONS = (
    ONE_GEO[: ONE_GEO.index("data {")].rstrip()[:-1]
    + """  vector mediated(matrix run1, matrix planned1, array[] int channel1, array[] int from1,
                  vector beta1, vector retention1, vector half1, vector slope1, matrix run2,
                  matrix planned2, array[] int channel2, array[] int from2, vector beta2,
                  vector retention2, vector half2, vector slope2, real gamma) {
    int D = size(channel1);
    int L1 = cols(run1);
    int L2 = cols(run2);
    vector[D] mean;
    for (d in 1:D) {
      int m = channel1[d];
      vector[L1] hr = saturate(adstock(run1[d]', retention1[m]), half1[m], slope1, m);
      vector[L1] hp = saturate(adstock(planned1[d]', retention1[m]), half1[m], slope1, m);
      mean[d] = gamma * beta1[m] * sum(hr[from1[d]:L1] - hp[from1[d]:L1]);
      if (channel2[d] > 0) {
        int k = channel2[d];
        vector[L2] gr = saturate(adstock(run2[d]', retention2[k]), half2[k], slope2, k);
        vector[L2] gp = saturate(adstock(planned2[d]', retention2[k]), half2[k], slope2, k);
        mean[d] += beta2[k] * sum(gr[from2[d]:L2] - gp[from2[d]:L2]);
      }
    }
    return mean;
  }
}
"""
)
STAGE_DATA = """  int<lower=1> L@;
  array[N] int<lower=1, upper=L@> window@;
  int<lower=0, upper=N> O@;
  array[O@] int<lower=1, upper=N> observed@;
  vector[O@] y@;
  int<lower=1> M@;
  matrix[L@, M@] spend@;
  vector<lower=0>[M@] window_spend@;
  vector[M@] roi_mu@;
  vector<lower=0>[M@] roi_sigma@;
  int<lower=0> P@;
  matrix[N, P@] calendar@;
  int<lower=0> C@;
  matrix[N, C@] controls@;
  vector[C@] control_lower@;
  vector[C@] control_upper@;
  int<lower=0, upper=1> tanh_curve@;
  int<lower=0> E@;
  array[E@] int<lower=1, upper=M@> estimate_channel@;
  matrix[E@, L@] as_run@;
  matrix[E@, L@] as_planned@;
  array[E@] int<lower=1, upper=L@> estimate_readout@;
  vector[E@] lift@;
  vector<lower=0>[E@] lift_se@;
"""
STAGE_PARAMETERS = """  vector[M@] log_roi@;
  vector<lower=0, upper=1>[M@] retention@;
  vector<lower=0>[M@] half@;
  vector<lower=0>[tanh_curve@ ? 0 : M@] slope@;
  real intercept@;
  vector[P@] calendar_coef@;
  vector<lower=control_lower@, upper=control_upper@>[C@] control_coef@;
  real<lower=0> sigma@;
"""
STAGE_MEAN = """  matrix[N, M@] h@ = saturated(spend@, window@, retention@, half@, slope@);
  vector[M@] beta@ = exp(log_roi@) .* window_spend@ ./ (rep_row_vector(1, N) * h@)';
  vector[N] mu@ = intercept@ + calendar@ * calendar_coef@ + controls@ * control_coef@ + h@ * beta@;
  vector[E@] l@ = estimated(as_run@, as_planned@, estimate_channel@, estimate_readout@, beta@,
                            retention@, half@, slope@);
"""
STAGE_MODEL = """  log_roi@ ~ normal(roi_mu@, roi_sigma@);
  retention@ ~ beta(1, 3);
  half@ ~ lognormal(0, 1);
  slope@ ~ lognormal(0, 0.5);
  intercept@ ~ normal(1, 1);
  calendar_coef@ ~ std_normal();
  control_coef@ ~ std_normal();
  sigma@ ~ std_normal();
  y@ ~ normal(mu@[observed@], sigma@);
  lift@ ~ normal(l@, lift_se@);
"""
MEDIATED = """  real gamma = control_coef2[mediator] / mediator_sd * mediator_unit;
  vector[D] m = mediated(run1, planned1, channel1, from1, beta1, retention1, half1, slope1, run2,
                         planned2, channel2, from2, beta2, retention2, half2, slope2, gamma);
"""


def stages(template: str) -> str:
    return template.replace("@", "1") + template.replace("@", "2")


PROGRAM = (
    FUNCTIONS
    + "data {\n  int<lower=1> N;\n"
    + stages(STAGE_DATA)
    + """  int<lower=1, upper=C2> mediator;
  real<lower=0> mediator_sd;
  real<lower=0> mediator_unit;
  int<lower=0> D;
  array[D] int<lower=1, upper=M1> channel1;
  matrix[D, L1] run1;
  matrix[D, L1] planned1;
  array[D] int<lower=1, upper=L1> from1;
  array[D] int<lower=0, upper=M2> channel2;
  matrix[D, L2] run2;
  matrix[D, L2] planned2;
  array[D] int<lower=1, upper=L2> from2;
  vector[D] mediated_lift;
  vector<lower=0>[D] mediated_se;
}
parameters {
"""
    + stages(STAGE_PARAMETERS)
    + "}\nmodel {\n"
    + stages(STAGE_MEAN)
    + MEDIATED
    + stages(STAGE_MODEL)
    + "  mediated_lift ~ normal(m, mediated_se);\n}\n"
    + "generated quantities {\n  vector[O1 + E1 + O2 + E2 + D] log_lik;\n  {\n"
    + stages(STAGE_MEAN).replace("\n  ", "\n    ").replace("  matrix", "    matrix", 1)
    + MEDIATED.replace("\n  ", "\n    ").replace("  real", "    real", 1)
    + """    for (o in 1:O1) {
      log_lik[o] = normal_lpdf(y1[o] | mu1[observed1[o]], sigma1);
    }
    for (e in 1:E1) {
      log_lik[O1 + e] = normal_lpdf(lift1[e] | l1[e], lift_se1[e]);
    }
    for (o in 1:O2) {
      log_lik[O1 + E1 + o] = normal_lpdf(y2[o] | mu2[observed2[o]], sigma2);
    }
    for (e in 1:E2) {
      log_lik[O1 + E1 + O2 + e] = normal_lpdf(lift2[e] | l2[e], lift_se2[e]);
    }
    for (d in 1:D) {
      log_lik[O1 + E1 + O2 + E2 + d] = normal_lpdf(mediated_lift[d] | m[d], mediated_se[d]);
    }
  }
}
"""
)
WEEKS, SEED, POINTS, CHAINS, WARMUP, DRAWS, BATCHES = 104, 20261005, 25, 4, 1000, 10_000, 10
CHANNELS = ("tv", "search")
GEO = Geo("national")
SEASON = FourierSeason(52.18, 1)
TRUTH = {
    "tv_on_searches": 0.5,
    "searches_on_sales": 2.0,
    "search_on_sales": 1.5,
    "tv_on_sales": 0.4,
    "tv": {"retention": 0.7, "half": 1.0, "slope": 1.0},
    "search": {"retention": 0.1, "half": 0.8, "slope": 1.0},
}
"""Television's return on searches, in searches a euro; a search's on sales, in euros; the search
ads' on sales and, in ``both``, television's on sales directly, in euros a euro; and each
channel's carryover and curve, its half-saturation point in units of its mean positive spend."""
WORLDS = ("mediated", "both")


def curve(spend: np.ndarray, channel: str, tanh: bool, length: int) -> np.ndarray:
    """A series over its channel's mean positive spend in the panel, carried over by the
    normalised kernel ``length`` long, then saturated."""
    shape = TRUTH[channel]
    w = shape["retention"] ** np.arange(length)
    a = np.convolve(np.r_[np.zeros(length - spend.size), spend], w / w.sum())[:length]
    if tanh:
        return np.tanh(math.atanh(0.5) * a / shape["half"])
    return a ** shape["slope"] / (a ** shape["slope"] + shape["half"] ** shape["slope"])


def specs(name: str) -> NestedSpec:
    both = name == "both"
    return NestedSpec(
        "searches",
        BayesSpec(
            (BayesMedia("tv", RoiPrior(0.0, 1.0)),),
            season=SEASON,
            saturation=TanhCurve() if both else HillCurve(),
        ),
        BayesSpec(
            ((BayesMedia("tv"),) if both else ()) + (BayesMedia("search"),),
            (Control("price", upper=0.0), Control("searches", lower=0.0)),
            season=SEASON,
        ),
    )


def world(name: str) -> tuple[MediaPanel, dict, dict, NestedSpec]:
    """The panel, the lifts on sales and the lifts on the searches."""
    both = name == "both"
    rng = np.random.default_rng([SEED, WORLDS.index(name)])
    t = np.arange(WEEKS)
    tv = rng.gamma(2.0, 50.0, WEEKS) * (rng.random(WEEKS) > 0.3)
    search = rng.gamma(2.0, 30.0, WEEKS) * (rng.random(WEEKS) > 0.2)
    unit = {"tv": tv[tv > 0].mean(), "search": search[search > 0].mean()}
    season = np.sin(2.0 * np.pi * t / 52.18)

    def contribution(spend: np.ndarray, channel: str, tanh: bool, roi: float) -> np.ndarray:
        h = curve(spend / unit[channel], channel, tanh, WEEKS)
        return roi * spend.sum() * h / h.sum()

    def lift(e: LiftEstimate, channel: str, tanh: bool, roi: float, panel_spend) -> float:
        """The group's lift at the world's parameters, the coefficient its panel return's."""
        h = curve(panel_spend / unit[channel], channel, tanh, WEEKS)
        beta = roi * panel_spend.sum() / h.sum()
        length = max(WEEKS, e.spend.size)
        run, planned = (
            curve(s / e.share / unit[channel], channel, tanh, length) for s in (e.spend, e.planned)
        )
        return e.share * beta * float((run - planned)[-e.readout :].sum())

    searches = (
        500.0
        + 50.0 * season
        + contribution(tv, "tv", both, TRUTH["tv_on_searches"])
        + 15.0 * rng.standard_normal(WEEKS)
    )
    price = 10.0 + 0.5 * rng.standard_normal(WEEKS)
    sales = (
        1000.0
        + 40.0 * season
        + TRUTH["searches_on_sales"] * searches
        + contribution(search, "search", False, TRUTH["search_on_sales"])
        + (contribution(tv, "tv", False, TRUTH["tv_on_sales"]) if both else 0.0)
        - 20.0 * (price - 10.0)
        + 20.0 * rng.standard_normal(WEEKS)
    )
    panel = MediaPanel(
        dates=np.datetime64("2023-01-02") + np.timedelta64(7, "D") * t,
        cadence=Cadence.WEEK,
        geos=(GEO,),
        channels=tuple(Channel(c) for c in CHANNELS),
        controls=("price", "searches"),
        kpi=sales[None, :],
        kpi_unit=Revenue(),
        spend=np.column_stack([tv, search])[None],
        currency="EUR",
        control_values=np.column_stack([price, searches])[None],
        filled=np.zeros((1, WEEKS), dtype=bool),
    )
    dark = (np.r_[np.full(4, 20.0), np.zeros(6)], np.full(10, 20.0), 6, 0.2)
    up = (
        (np.r_[np.full(WEEKS + 4, 25.0), np.full(6, 50.0)], np.full(WEEKS + 10, 25.0), 6, 0.25)
        if not both
        else (np.r_[np.full(6, 100.0), np.full(4, 200.0)], np.full(10, 100.0), 4, 1.0)
    )
    on_sales = []
    for spend, planned, readout, share in (dark, up):
        e = LiftEstimate(spend, planned, readout, 0.0, 1.0, share)
        mean = TRUTH["searches_on_sales"] * lift(e, "tv", both, TRUTH["tv_on_searches"], tv)
        if both:
            mean += lift(e, "tv", False, TRUTH["tv_on_sales"], tv)
        se = 0.3 * abs(mean)
        on_sales.append(
            LiftEstimate(spend, planned, readout, mean + se * rng.standard_normal(), se, share)
        )
    heavy = LiftEstimate(
        np.r_[np.full(4, 5.0), np.full(4, 10.0)], np.full(8, 5.0), 4, 0.0, 1.0, 0.1
    )
    mean = lift(heavy, "search", False, TRUTH["search_on_sales"], search)
    estimates = {
        "tv": on_sales,
        "search": [
            LiftEstimate(
                heavy.spend, heavy.planned, 4, mean + 4.0 * rng.standard_normal(), 4.0, 0.1
            )
        ],
    }
    mean = lift(LiftEstimate(*dark[:3], 0.0, 1.0, 0.2), "tv", both, TRUTH["tv_on_searches"], tv)
    on_searches = {
        "tv": [LiftEstimate(dark[0], dark[1], 6, mean + 15.0 * rng.standard_normal(), 15.0, 0.2)]
    }
    return panel, estimates, on_searches, specs(name)


def stage_data(model, panel_kpi: np.ndarray, spend: np.ndarray, estimates, suffix: str) -> dict:
    """One stage's data on its model's units, suffixed: the spend and the estimates laid out from
    the panel and the estimates themselves, the rest read off the model."""
    names = [m.channel for m in model.spec.media]
    length, weeks = model.lagged.shape[2], spend.shape[0]
    spend = spend[:, [CHANNELS.index(c) for c in names]]
    read = []
    for j, c in enumerate(names):
        for e in estimates.get(c, []):
            series = [
                np.r_[np.zeros(length - s.size), s / e.share / model.spend_unit[j]]
                for s in (e.spend, e.planned)
            ]
            read.append((j, *series, length - e.readout + 1, e.lift / e.share, e.se / e.share))
    data = {
        "L": length,
        "window": list(range(length - weeks + 1, length + 1)),
        "O": int(model.observed.sum()),
        "observed": (np.flatnonzero(model.observed) + 1).tolist(),
        "y": (panel_kpi[model.observed] / model.kpi_unit).tolist(),
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
        "tanh_curve": int(isinstance(model.spec.saturation, TanhCurve)),
        "E": len(read),
        "estimate_channel": [r[0] + 1 for r in read],
        "as_run": [r[1].tolist() for r in read],
        "as_planned": [r[2].tolist() for r in read],
        "estimate_readout": [r[3] for r in read],
        "lift": [r[4] / model.kpi_unit for r in read],
        "lift_se": [r[5] / model.kpi_unit for r in read],
    }
    return {f"{key}{suffix}": value for key, value in data.items()}


def data(model, panel: MediaPanel, estimates, on_searches, spec: NestedSpec) -> dict:
    one, two = model.stage_one, model.stage_two
    staged = [m.channel for m in spec.stage_one.media]
    direct = [m.channel for m in spec.stage_two.media]
    k = [c.column for c in spec.stage_two.controls].index(spec.mediator)
    rows = []
    for c in staged:
        for e in estimates.get(c, []):
            j1 = staged.index(c)
            j2 = direct.index(c) if c in direct else None
            first = [
                np.r_[np.zeros(one.lagged.shape[2] - s.size), s / e.share / one.spend_unit[j1]]
                for s in (e.spend, e.planned)
            ]
            second = [
                np.r_[np.zeros(two.lagged.shape[2] - s.size), s / e.share / two.spend_unit[j2]]
                if j2 is not None
                else np.zeros(two.lagged.shape[2])
                for s in (e.spend, e.planned)
            ]
            rows.append(
                (
                    j1 + 1,
                    *first,
                    one.lagged.shape[2] - e.readout + 1,
                    0 if j2 is None else j2 + 1,
                    *second,
                    two.lagged.shape[2] - e.readout + 1,
                    e.lift / e.share / two.kpi_unit,
                    e.se / e.share / two.kpi_unit,
                )
            )
    searches = panel.control_values[0, :, panel.controls.index(spec.mediator)]
    return {
        "N": panel.dates.size,
        **stage_data(one, searches, panel.spend[0], on_searches, "1"),
        **stage_data(
            two,
            panel.kpi[0],
            panel.spend[0],
            {c: es for c, es in estimates.items() if c not in staged},
            "2",
        ),
        "mediator": k + 1,
        "mediator_sd": float(two.control_sd[k]),
        "mediator_unit": one.kpi_unit,
        "D": len(rows),
        "channel1": [r[0] for r in rows],
        "run1": [r[1].tolist() for r in rows],
        "planned1": [r[2].tolist() for r in rows],
        "from1": [r[3] for r in rows],
        "channel2": [r[4] for r in rows],
        "run2": [r[5].tolist() for r in rows],
        "planned2": [r[6].tolist() for r in rows],
        "from2": [r[7] for r in rows],
        "mediated_lift": [r[8] for r in rows],
        "mediated_se": [r[9] for r in rows],
    }


def unconstrained(header: list[str], draws: np.ndarray, model, suffix: str) -> np.ndarray:
    """Stan's draws of one stage on mixdyn's sampling coordinates, the inverse of each block's
    transform."""

    def block(prefix: str) -> np.ndarray:
        name = prefix + suffix
        names = [h for h in header if h == name or h.startswith(name + ".")]
        return draws[:, [header.index(h) for h in names]]

    signs = np.array([c.lower != -np.inf or c.upper != np.inf for c in model.spec.controls])
    control = block("control_coef")
    theta = np.column_stack(
        [
            block("log_roi"),
            np.log(block("retention") / (1.0 - block("retention"))),
            np.log(block("half")),
            np.log(block("slope")),
            block("intercept"),
            block("calendar_coef"),
            np.where(signs, np.log(np.abs(np.where(signs, control, 1.0))), control),
            np.log(block("sigma")),
        ]
    )
    if theta.shape[1] != model.dimension:
        msg = (
            f"Stan's draws give {theta.shape[1]} coordinates; stage {suffix} has {model.dimension}"
        )
        raise ValueError(msg)
    return theta


def recorded(estimates: dict) -> dict:
    return {
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
    }


def case(executable: Path, work: Path, name: str) -> dict[str, object]:
    panel, estimates, on_searches, spec = world(name)
    model = nested_model(panel, GEO, spec, estimates=estimates, mediator_estimates=on_searches)
    (work / f"{name}.data.json").write_text(
        json.dumps(data(model, panel, estimates, on_searches, spec))
    )
    points = np.array(
        [
            model._lift(x)
            for x in np.random.default_rng([SEED, WORLDS.index(name), 1]).uniform(
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
    stan(
        executable,
        work,
        name,
        ["sample", f"num_samples={DRAWS}", f"num_warmup={WARMUP}", f"num_chains={CHAINS}",
         "random", f"seed={SEED}", f"num_threads={CHAINS}"],
        "sample",
    )  # fmt: skip
    header, chains = None, []
    for c in range(1, CHAINS + 1):
        header, values = csv(work / f"{name}.sample_{c}.csv")
        chains.append(values)
    draws = np.array(chains).reshape(CHAINS * DRAWS, -1)
    one, two = model.stage_one, model.stage_two
    theta = np.column_stack(
        [unconstrained(header, draws, one, "1"), unconstrained(header, draws, two, "2")]
    )
    posterior = {
        stage: {
            key: summary(values.reshape(CHAINS, DRAWS, -1))
            for key, values in m.parameters(part).items()
        }
        for stage, m, part in (
            ("stage_one", one, theta[:, : one.dimension]),
            ("stage_two", two, theta[:, one.dimension :]),
        )
    }
    divergent = int(draws[:, header.index("divergent__")].sum())
    log_lik = draws[:, [i for i, h in enumerate(header) if h.startswith("log_lik.")]]
    spread = np.linspace(0, CHAINS * DRAWS - 1, POINTS).round().astype(int)
    return {
        "panel": {
            "kpi": panel.kpi[0].tolist(),
            "spend": panel.spend[0].tolist(),
            "controls": panel.control_values[0].tolist(),
        },
        "estimates": recorded(estimates),
        "mediator_estimates": recorded(on_searches),
        "stage_one_saturation": type(spec.stage_one.saturation).__name__,
        "stage_two_media": [m.channel for m in spec.stage_two.media],
        "points": points.tolist(),
        "log_prob": log_prob[:, 0].tolist(),
        "gradient": log_prob[:, 1:].tolist(),
        "posterior": posterior,
        "log_lik": summary(log_lik.reshape(CHAINS, DRAWS, -1)),
        "log_lik_draws": {"theta": theta[spread].tolist(), "log_lik": log_lik[spread].tolist()},
        "divergent": divergent,
    }


def main() -> None:
    cmdstan, work, fixture = (Path(a).resolve() for a in sys.argv[1:4])
    work.mkdir(parents=True, exist_ok=True)
    source = work / "nested.stan"
    if not source.exists() or source.read_text() != PROGRAM:
        source.write_text(PROGRAM)
    executable = work / "nested"
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
                    "generator": "CmdStan's log_prob and sample on `program`, two nested worlds",
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
