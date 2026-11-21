---
name: mixdyn-media-mix
description: Fit, check and plan a media mix with mixdyn, one gate at each step. Use when asked to build or run a media-mix model (MMM) on a weekly panel of a KPI and media spend, to read a lift or geo test into one, or to recommend a budget split from one. Walks read, fit, diagnostics, tests, plan and report, and stops where the engine's own checks fail.
---

<!-- --8<-- [start:walkthrough] -->
# A media mix with mixdyn, gated at every step

**Fitted is not approved.** A model that fits the KPI has not shown that its returns are causal:
channels bought together, or bought when demand rises, fit as well under wrong returns as under
true ones. Each step below ends in a gate, the engine's own check. A failed gate stops the work
and goes into the report; it is never worked around. Only a lift or geo test the model did not
fit to shows that a return is right.

The blocks run in order in one Python session (`tests/test_skill.py` runs them on the stand-in
below). The engine runs in double precision. The decisions the steps rest on are in `docs/adr/`;
the public names are in `docs/api/index.md`.

## 1. Read the panel

One row per period, and per geo where there are several; the dates spaced as the cadence says
(data finer than the cadence is summed to it first); each channel's spend in one currency.

```python
import jax
import numpy as np

from mixdyn import NATIONAL, Cadence, Revenue, read_panel

jax.config.update("jax_enable_x64", True)

# A stand-in for the user's data: two years of weekly revenue and two channels' spend.
rng = np.random.default_rng(0)
weeks = 104
tv, search = rng.uniform(50.0, 150.0, weeks), rng.uniform(20.0, 60.0, weeks)
data = {
    "week": np.datetime64("2024-01-01") + np.timedelta64(7, "D") * np.arange(weeks),
    "revenue": 2000.0 + 4.0 * tv + 6.0 * search + rng.normal(0.0, 40.0, weeks),
    "tv": tv,
    "search": search,
}
panel = read_panel(
    data,
    date="week",
    kpi="revenue",
    spend={"tv": "tv", "search": "search"},
    cadence=Cadence.WEEK,
    kpi_unit=Revenue(),
    currency="EUR",
)
for issue in panel.issues:
    print(issue.severity, issue.code, issue.message)
```

**Gate.** `read_panel` raises `PanelError` listing every error; fix the data, never the check.
Every warning in `panel.issues` goes into the report: `FEW_OBSERVATIONS`, fewer than 15 observed
periods per effect, means the effects lean on their priors.

## 2. Fit

The Bayesian engine (ADRs 0010, 0011): each channel's return, carryover and curve, with a
return prior a buyer can state. Its defaults are four chains of 1000 draws after 1000 of warm-up;
the example draws fewer to stay short.

```python
from mixdyn.models.bayes_mmm import BayesMedia, BayesSpec, fit_bayes

spec = BayesSpec((BayesMedia("tv"), BayesMedia("search")))
fit = fit_bayes(panel, NATIONAL, spec, seed=0, chains=2, warmup=300, draws=300)
```

**Gate.** None yet: a fit that returns has only run.

## 3. Diagnostics: the health card

The health card (ADR 0060) reads the largest R-hat, a posterior predictive p-value of the KPI's
total, the probability of a negative baseline, and how far each channel's return moved from its
prior. Its thresholds are the caller's; Meridian's model reviewer reads the three below.

```python
from mixdyn.inference.diagnostics import health

card = health(fit, panel, NATIONAL, seed=0, prior_draws=1000)
flags = card.flags(max_rhat=1.2, min_ppp=0.05, max_negative_baseline=0.2)
print(f"R-hat {card.max_rhat:.3f}, ppp {card.ppp:.2f}, baseline < 0 {card.negative_baseline:.2f}")
print("posterior sd over prior sd", {c: round(w, 2) for c, w in card.width.items()})
if not all(flags.values()):
    raise SystemExit(f"the fit fails its health card, {flags}: report it and stop")
```

**Gate.** A failed flag stops the work: the report states the cell and its value, and no plan is
made from the fit. A return whose posterior is about as wide as its prior (`card.width` near 1)
was read from the prior, not the data: report it as such.

## 4. Tests

A lift or geo test is the only evidence that reads a return without the model's assumptions.
Read each test into a `LiftEstimate`: its spend as run and as planned, its readout, its lift and
standard error, its share of the market and the date its readout ends. A geo test is read by the
engine's readers (`read_did`, `read_tbr`, `read_synthetic_control`, `read_sdid`; ADRs 0047,
0048), a switchback by `read_switchback` (ADR 0056). Then refit with `estimates=tests` and go back
to step 3.

```python
from mixdyn.api import estimands

tests = {}  # each channel's tests, LiftEstimate by LiftEstimate; empty: none has been run
for estimand in estimands(panel, tests):
    print(estimand.estimand)
```

**Gate.** Each test's estimand is stated before it enters a fit: a lift against zero spend, or a
marginal change between two levels, which says nothing of the curve below the lower (ADR 0046).
No test means the plan in step 5 is the no-test plan, and the report recommends a test.

## 5. Plan

With no test, the plan is the no-test safety clause (ADR 0044): the split that does best over the
plan in place in the worst tenth of the posterior's draws, `plan_cvar` at level 0.1, so it moves
only as far as the evidence does and is never worse than the plan in place there. With tests read,
`plan_posterior` plans for the posterior's expected return (ADRs 0019, 0034).

```python
from mixdyn.decisions import Range, plan_cvar, plan_posterior, readings

periods = 13
current = panel.spend[0, -periods:].mean(axis=0)  # the plan in place, a week
budget = float(current.sum()) * periods
limits = {c: Range(0.5 * s, 2.0 * s) for c, s in zip(panel.channels, current, strict=True)}
if tests:
    plan = plan_posterior(panel, NATIONAL, readings(fit), budget, periods, limits)
else:
    plan = plan_cvar(
        panel, NATIONAL, readings(fit), budget, periods, limits, against=current, level=0.1
    )
print({c: round(float(s), 1) for c, s in zip(plan.channels, plan.spend, strict=True)})
```

**Gate.** The plan stays inside the limits the user set and spends the budget; a limit the plan
sits on is reported as binding, since the data would move the plan past it.

## 6. Report

```python
report = {
    "panel warnings": [str(issue.code) for issue in panel.issues],
    "health card": flags,
    "tests read": [e.estimand for e in estimands(panel, tests)],
    "plan, a week": {c: round(float(s), 1) for c, s in zip(plan.channels, plan.spend, strict=True)},
    "approved": bool(tests),
}
print(report)
```

**Gate: the claims.** The report says what the engine measured and no more:

- the plan's gain over the plan in place as the posterior reads it, with its interval (for the
  no-test plan, `plan.cvar` is the mean gain in the worst tenth of the draws);
- the health card's cells at the thresholds named, and every panel warning;
- `"approved": False` unless a test the fit read stands behind the returns the plan moves money
  on. Without one, the plan is a candidate to test, and the report names the test to run first
  (`mixdyn.experiments.value_of_tests`, ADR 0031).

It never says the model is validated, causal or better than another tool. Whether the engine
beats another tool is read only by the scorecard's claim on a pre-registered run (ADR 0040), never
from one fit.
