# mixdyn

[![ci](https://github.com/causaldyn/mixdyn/actions/workflows/ci.yml/badge.svg)](https://github.com/causaldyn/mixdyn/actions/workflows/ci.yml)
[![pypi](https://img.shields.io/pypi/v/mixdyn)](https://pypi.org/project/mixdyn/)
[![python](https://img.shields.io/pypi/pyversions/mixdyn)](https://pypi.org/project/mixdyn/)
[![license](https://img.shields.io/pypi/l/mixdyn)](https://github.com/causaldyn/mixdyn/blob/main/LICENSE)
[![docs](https://img.shields.io/website?url=https%3A%2F%2Fcausaldyn.github.io%2Fmixdyn%2F&label=docs)](https://causaldyn.github.io/mixdyn/)
[![scorecard](https://api.scorecard.dev/projects/github.com/causaldyn/mixdyn/badge)](https://scorecard.dev/viewer/?uri=github.com/causaldyn/mixdyn)
[![doi](https://zenodo.org/badge/1466004118.svg)](https://doi.org/10.5281/zenodo.24425629)

<!-- --8<-- [start:intro] -->
**Media-mix modelling that plans the budget, reads your lift and geo tests, and is scored in the
open against Meridian, PyMC-Marketing and Orbit.**

mixdyn is a media-mix modelling (MMM) engine in Python and JAX, built on
[causal-hybrid-control](https://github.com/causaldyn/causal-hybrid-control):

- **Models**: regressions, discount dynamic linear models, Bayesian models with lift and geo tests
  in the likelihood, and hierarchies over geos.
- **Experiments**: readers for lift and geo tests, and a test priced by what it adds to the plan
  before it runs.
- **Plans**: next quarter's budget split over channels, inside each channel's bounds, from a
  posterior or from each channel's margin.
- **Checks**: diagnostics, holdout scores, refreshes as weeks arrive, and each channel's
  sensitivity to an unmeasured confounder.

Its runtime is NumPy, SciPy, JAX and causal-hybrid-control, with what those bring in: no
probabilistic-programming language, no other MMM package, no UI. It does not do multi-touch
attribution (MTA): it reads no user-level path.
<!-- --8<-- [end:intro] -->

## How it measures up

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/causaldyn/mixdyn/v0.2.3/docs/assets/closest-rival-dark.svg">
  <img alt="For each family and rung of three pre-registered runs, mixdyn's regret per euro less its closest rival's, with 95 % intervals: every interval but one, family 4's with no lift test, lies left of the margin δ = 0.05." src="https://raw.githubusercontent.com/causaldyn/mixdyn/v0.2.3/docs/assets/closest-rival-light.svg">
</picture>

<!-- --8<-- [start:measured] -->
On the [causaldyn bench](https://github.com/causaldyn/causaldyn-bench)'s simulated worlds, whose
optimal budget split is known, mixdyn's plans are within δ = 0.05 of regret per euro of budget of
those of Meridian 2.1.0, PyMC-Marketing 1.3.0 and Orbit 1.1.5.1, and of the status quo and the
equal split, on 37 of 38 pre-registered family-rungs. Three scored runs read them, M3, M4 and M5,
each pre-registration frozen before any confirmatory world was fitted. The one family-rung not
within δ is family 4, S-curves, with no lift test: there mixdyn's plan loses to Orbit's by +0.136
and to PyMC-Marketing's by +0.135 a euro, because the data do not identify television's curve
beyond the range its spend was logged in.

<!-- record:scoreboard -->
| Of 38 family-rungs, mixdyn | Meridian | PyMC-Marketing | Orbit | Status quo | Equal split |
|---|--:|--:|--:|--:|--:|
| beats the rival | 34 | 19 | 35 | 36 | 38 |
| ties | 1 | 11 | 1 | 2 | 0 |
| loses to it, within δ | 1 | 7 | 1 | 0 | 0 |
| loses to it, past δ | 0 | 1 | 1 | 0 | 0 |
| is not compared: the rival failed | 2 | 0 | 0 | 0 | 0 |
<!-- /record:scoreboard -->

Each count is of family-rungs, each read by Hall's reader at one-sided 0.025 over the worlds both
tools planned. This is a non-inferiority result on simulated worlds: it does not say that mixdyn is
the best MMM, and no real data has scored it yet. One maintainer built both the engine and the
bench that scores it. Every comparison, every loss, the pre-registrations, the versions and the
compute are on the [benchmark page](https://causaldyn.github.io/mixdyn/benchmark/).
<!-- --8<-- [end:measured] -->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/causaldyn/mixdyn/v0.2.3/docs/assets/cost-dark.svg">
  <img alt="Mean CPU time of a process and the largest peak memory of any, over M4 and M5: mixdyn's two arms about 10 CPU seconds and 0.18 GiB, Orbit, PyMC-Marketing and Meridian 75 to 245 seconds and up to 12.3 GiB." src="https://raw.githubusercontent.com/causaldyn/mixdyn/v0.2.3/docs/assets/cost-light.svg">
</picture>

## Install

<!-- --8<-- [start:install] -->
```bash
pip install mixdyn        # Python 3.12 or later
```
<!-- --8<-- [end:install] -->

## A first plan

<!-- --8<-- [start:example] -->
```python
import numpy as np

from mixdyn.decisions.margin_plan import plan_corner, shrink_margins, time_held
from mixdyn.models.margin_mmm import NoiseMixture, adstock, fit_margins

# A stand-in for two years of weekly history: three channels' spend, and sales that each
# channel's spend lifts through a geometric adstock (retention 0.5, eight weeks).
rng = np.random.default_rng(0)
weeks = 104
spend = rng.uniform([60.0, 20.0, 15.0], [180.0, 80.0, 55.0], size=(weeks, 3))
carried = np.column_stack([adstock(s, 0.5, 8, s[:13].mean()) for s in spend.T])
truth = np.array([1.5, 3.0, 0.5])  # each channel's margin, which the fit does not see
sales = 5000.0 + carried @ truth + rng.normal(0.0, 25.0, weeks)

# Each channel's margin, the sales one unit of weekly spend returns over the kernel, read under a
# mixture of error models, then shrunk towards the channels' common level.
fit = fit_margins(sales, spend, promotion=None, price=None, kernel=8, noise=NoiseMixture())
shrunk = shrink_margins(fit.mean, fit.cov)
_, held = time_held(spend)  # a channel whose spend follows the calendar stays where it is

# Next quarter's weekly plan at the status quo's budget, each channel between half and twice its
# status quo: every floor first, then the rest to the highest shrunk margin.
status_quo = spend[-13:].mean(axis=0)
low = np.where(held, status_quo, 0.5 * status_quo)
high = np.where(held, status_quo, 2.0 * status_quo)
plan = plan_corner(shrunk.mean, low, high, float(status_quo.sum()))
print("margins", fit.mean.round(2), "+/-", np.sqrt(np.diag(fit.cov)).round(2))
print("status quo", status_quo.round(1), "plan", plan.round(1))
```
<!-- --8<-- [end:example] -->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/causaldyn/mixdyn/v0.2.3/docs/assets/first-plan-dark.svg">
  <img alt="The example's three fitted margins with 95 % intervals beside the true ones, and its plan beside the status quo, inside half to twice each channel's status quo." src="https://raw.githubusercontent.com/causaldyn/mixdyn/v0.2.3/docs/assets/first-plan-light.svg">
</picture>

<!-- --8<-- [start:example-notes] -->
This is the linear form of `margin held`, the rule the runs scored with no lift test, which adds a
saturating curve to each channel and fills the budget along the curves (`scorecard/arms/margin.py`).
These functions sit at module paths, which are not public names yet: before 1.0 they can change in
any minor release. `SKILL.md` walks one panel through the Bayesian engine, gated at each step:
read, fit, diagnostics, tests, plan and report.
<!-- --8<-- [end:example-notes] -->

## Learn more

- [Documentation](https://causaldyn.github.io/mixdyn/): the walkthrough, the scope, and the API
  reference with its versioning rules.
- [Benchmark](https://causaldyn.github.io/mixdyn/benchmark/): the full record of M3, M4 and M5,
  every table generated from the runs' records (`docs/benchmark.md`).
- `evidence/`: the scripts and records the decisions' measurements come from.

## Working on it

```bash
just sync    # .venv exactly as uv.lock has it
just check   # format, lint, types, the dependency gate, tests: what CI runs
```

Runtime packages are limited to `allowed-runtime-packages.txt`, and CI checks every push against
it. `docs/adr/` records the decisions the code relies on. `CONTRIBUTING.md` says how a change is
made and signed off, and `CHANGELOG.md` lists each release's changes.

## Licence

Apache-2.0 (`LICENSE`). Every redistribution carries `NOTICE`.
