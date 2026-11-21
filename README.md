# mixdyn

[![ci](https://github.com/causaldyn/mixdyn/actions/workflows/ci.yml/badge.svg)](https://github.com/causaldyn/mixdyn/actions/workflows/ci.yml)
[![pypi](https://img.shields.io/pypi/v/mixdyn)](https://pypi.org/project/mixdyn/)
[![python](https://img.shields.io/pypi/pyversions/mixdyn)](https://pypi.org/project/mixdyn/)
[![license](https://img.shields.io/pypi/l/mixdyn)](https://github.com/causaldyn/mixdyn/blob/main/LICENSE)
[![docs](https://img.shields.io/website?url=https%3A%2F%2Fcausaldyn.github.io%2Fmixdyn%2F&label=docs)](https://causaldyn.github.io/mixdyn/)
[![scorecard](https://api.scorecard.dev/projects/github.com/causaldyn/mixdyn/badge)](https://scorecard.dev/viewer/?uri=github.com/causaldyn/mixdyn)
[![doi](https://zenodo.org/badge/1466004118.svg)](https://doi.org/10.5281/zenodo.24425629)

<!-- --8<-- [start:intro] -->
A media-mix modelling (MMM) engine in Python and JAX: regressions, discount dynamic linear models,
Bayesian models with lift and geo tests in the likelihood, geo hierarchies, test readers and budget
plans, computed without a UI. Built on
[causal-hybrid-control](https://github.com/causaldyn/causal-hybrid-control). Apache-2.0.

What it has been measured to do, and where it loses, is below: one pre-registered run of the
[causaldyn bench](https://github.com/causaldyn/causaldyn-bench), stated with its losses. Nothing
here says "best".

**Where it sits.** Marketers' measurement is turning to MMM checked by experiments, more than to
multi-touch attribution (MTA), which follows each user's path to a sale. In a survey of 196 US
marketers that EMARKETER fielded for TransUnion in July 2025, 47 % planned to raise their spend on
MMM in the next year and 35 % their spend on MTA
([TransUnion, October 2025](https://newsroom.transunion.com/new-transunion-research-reveals-marketers-confidence-in-measurement-has-stalled/)).
Forrester's 2025 and 2026 global marketing surveys found that "adoption of marketing mix
modeling and incrementality testing increased substantially year over year"
([Forrester, July 2026](https://www.forrester.com/blogs/ai-moves-marketing-measurement-from-insights-to-action/)).
CIMM's 2026 roadmap for MMM asks the industry to "establish an experimentation and validation
playbook for geo tests, lift tests"
([CIMM, July 2026](https://cimm-us.org/cimm-outlines-industry-roadmap-to-strengthen-marketing-mix-modeling-as-ai-accelerates-media-decision-making/)).
mixdyn is built for that pairing: its Bayesian models read lift and geo tests in the likelihood,
and `mixdyn.experiments` prices a test by what it adds to the plan before it runs (ADRs 0030 to
0032). It does not do MTA: it reads no user-level path.

## Install

```bash
pip install mixdyn        # Python 3.12 or later
```

The runtime packages are NumPy, SciPy, JAX and causal-hybrid-control, with what those bring in: no
probabilistic-programming language and no other MMM package (ADR 0001).

<!-- --8<-- [end:intro] -->
<!-- --8<-- [start:record] -->
## A first plan

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
sales = 5000.0 + carried @ np.array([1.5, 3.0, 0.5]) + rng.normal(0.0, 25.0, weeks)

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

This is the linear form of the rule the measured run used with no lift test (ADRs 0086, 0091 and
0092). The arm that run scored, `margin held`, adds a saturating curve to each channel and fills
the budget along the curves; `scorecard/arms/margin.py` assembles it from these functions. These
functions sit at module paths, which are not public names yet: before 1.0 they can change in any
minor release (`docs/api/index.md`).

`SKILL.md` walks one panel through the Bayesian engine, gated at each step: read, fit,
diagnostics, tests, plan and report.

## Conflict of interest

One maintainer built both the engine and the bench that scores it: the bench's worlds, its regret
axis and the scripts that run each rival. The same maintainer wrote this run's pre-registration and
ran the run. No one independent of the engine has run it again. What limits the conflict: the
pre-registration was frozen before any confirmatory world was fitted (with no public time-stamp,
below); each rival ran in its own pinned environment, at its recommended settings and at the
setting its own tuning criterion chose; and the bench and the harness are public. What does not
limit it: the choice of the families, of the axis and of the rivals' scripts.

## What has been measured

**The claim.** On the causaldyn bench's simulated worlds, whose truth is known, mixdyn's plans are
non-inferior to those of Meridian 2.1.0, PyMC-Marketing 1.3.0 and Orbit 1.1.5.1, and to the status
quo and the equal split, within δ = 0.05 of regret per euro of budget, on 17 pre-registered
family-rungs. Each of the 17 verdicts was tested at one-sided 0.025 in a fixed sequence, on fresh
confirmatory seeds that no development step drew. All 17 held; the pre-registration expected
16.94.

This is a non-inferiority claim. It does not say that mixdyn is the best MMM, and it does not say
that mixdyn is within δ of every maintained MMM (see "No headline" below).

- **The axis.** The regret of the quarter's budget split a tool recommends, per euro of budget: the
  KPI the best split earns and the recommended split does not. Lower is better.
- **A comparison** is mixdyn's arm less one rival, paired by world, over the worlds both planned.
  It is within δ where Hall's (1992) skewness-corrected one-look reader shows the mean below 0.05
  at one-sided 0.025. A verdict is one family at one rung: an intersection-union test over its
  comparisons.
- **The rungs.** k = 0 reads the history alone. The tested rung reads the history and four go-dark
  lift tests (family 7: one test on 52 weeks of history, two on 78 and 104 weeks).
- **The arms.** At k = 0, `margin held`: the margin rule above, with its curves and its time guard.
  At the tested rung, `gs held`: it plays `anchored held`, margins anchored on the lift tests, and
  routes a world that `anchored held` flags to the Bayesian engine's `stacked` plan (ADRs 0088,
  0089 and 0093). On the geo families, `geo eb`, a closed-form model over geos fitted by empirical
  Bayes (ADR 0090).

Each cell is mixdyn's arm less the rival, in regret per euro: the mean over the worlds both
planned, with its paired 95 % Student-t interval. A negative cell is a lower regret than the
rival's.

| Family | Rung | Arm | Worlds | Status quo | Equal split | Meridian | PyMC-Marketing | Orbit |
|---|---|---|---|---|---|---|---|---|
| 0 Track M v2 | 0 | `margin held` | 350 | −0.128 [−0.147, −0.109] | −0.319 [−0.352, −0.287] | −0.051 [−0.072, −0.031] | −0.148 [−0.175, −0.120] | −0.182 [−0.215, −0.149] |
| 0 Track M v2 | 4 | `gs held` | 350 | −0.242 [−0.257, −0.228] | −0.433 [−0.466, −0.400] | −0.076 [−0.085, −0.068] | +0.001 [−0.005, +0.006] | −0.064 [−0.077, −0.051] |
| 2 Confounded spend | 0 | `margin held` | 240 | −0.090 [−0.114, −0.066] | −0.086 [−0.112, −0.060] | −0.044 [−0.069, −0.019] | −0.113 [−0.146, −0.079] | −0.100 [−0.136, −0.063] |
| 2 Confounded spend | 4 | `gs held` | 240 | −0.171 [−0.193, −0.147] | −0.166 [−0.191, −0.141] | −0.068 [−0.082, −0.054] | +0.005 [−0.003, +0.013] | −0.036 [−0.051, −0.021] |
| 3 Collinear channels | 0 | `margin held` | 270 | −0.059 [−0.080, −0.038] | −0.042 [−0.064, −0.021] | −0.021 [−0.040, −0.002] | −0.100 [−0.135, −0.065] | −0.103 [−0.137, −0.068] |
| 3 Collinear channels | 4 | `gs held` | 270 | −0.181 [−0.201, −0.161] | −0.165 [−0.188, −0.142] | −0.082 [−0.093, −0.070] | +0.008 [+0.003, +0.013] | −0.065 [−0.081, −0.049] |
| 5 Budget follows performance | 0 | `margin held` | 240 | −0.077 [−0.100, −0.055] | −0.066 [−0.086, −0.045] | −0.042 [−0.062, −0.022] | −0.109 [−0.143, −0.074] | −0.138 [−0.172, −0.104] |
| 5 Budget follows performance | 4 | `gs held` | 240 | −0.158 [−0.179, −0.137] | −0.146 [−0.172, −0.121] | −0.069 [−0.083, −0.054] | +0.020 [+0.010, +0.030] | −0.056 [−0.075, −0.036] |
| 7 Short series | 0 | `margin held` | 240 | −0.075 [−0.099, −0.051] | −0.043 [−0.063, −0.022] | −0.045 [−0.066, −0.024] | −0.108 [−0.144, −0.073] | −0.127 [−0.163, −0.091] |
| 7 Short series | 1–2 | `gs held` | 240 | −0.153 [−0.177, −0.129] | −0.122 [−0.147, −0.096] | −0.062 [−0.078, −0.046] | −0.007 [−0.020, +0.006] | −0.036 [−0.052, −0.019] |
| 13 PyMC-Marketing's priors | 4 | `gs held` | 240 | −0.072 [−0.102, −0.043] | +0.000 [−0.029, +0.029] | −0.025 [−0.046, −0.004] | −0.005 [−0.020, +0.011] | −0.134 [−0.165, −0.102] |
| 8 Geo | 0 | `geo eb` | 240 | −0.077 [−0.082, −0.071] | −0.041 [−0.045, −0.037] | −0.043 [−0.046, −0.039] | −0.008 [−0.010, −0.006] | −0.068 [−0.072, −0.063] |
| 8 Geo | 4 | `geo eb` | 240 | −0.077 [−0.082, −0.071] | −0.041 [−0.046, −0.037] | −0.044 [−0.048, −0.041] | −0.008 [−0.010, −0.006] | −0.056 [−0.060, −0.051] |
| 9 Spillover between geos | 0 | `geo eb` | 240 | −0.112 [−0.122, −0.102] | −0.053 [−0.060, −0.046] | −0.048 [−0.054, −0.042] | +0.000 [−0.003, +0.004] | −0.057 [−0.063, −0.050] |
| 9 Spillover between geos | 4 | `geo eb` | 240 | −0.106 [−0.116, −0.096] | −0.047 [−0.054, −0.040] | −0.044 [−0.051, −0.038] | +0.005 [+0.002, +0.009] | −0.061 [−0.071, −0.052] |
| 10 Few geos | 0 | `geo eb` | 240 | −0.069 [−0.076, −0.062] | −0.038 [−0.043, −0.033] | −0.031 [−0.035, −0.026], capped | −0.003 [−0.005, −0.000] | −0.043 [−0.049, −0.037] |
| 10 Few geos | 4 | `geo eb` | 240 | −0.069 [−0.076, −0.062] | −0.038 [−0.043, −0.032] | −0.034 [−0.039, −0.029], capped | −0.002 [−0.004, +0.001] | −0.038 [−0.044, −0.032] |

"Worlds" counts the confirmatory worlds the family declared. A failed fit leaves its world out of
the comparisons that read it, and is counted: `margin held` failed on 1 world of family 7 at k = 0
(47 weeks of history cannot identify its 19 nuisance columns), and PyMC-Marketing on 49 worlds of
families 8, 9, 10 and 15. On family 10, Meridian is declared capped, as in the run before, so its
comparisons are exploratory: it ran on every world, and it is read beside that verdict, not in it.

**The error bars.** Each interval is the paired 95 % Student-t interval of the mean difference over
a cell's complete cases: `mean ± t(0.975, n − 1) · s / √n`, where `n` is the number of complete
cases and `s` the standard deviation of the per-world differences, with divisor `n − 1`. The
verdict's word and level are Hall's reader's; the interval printed is Student's. The variability
is from world to world. A world is the bench's draw from one seed of one of the family's
environments, which draws what the environment leaves free: on Track M v2's `reference` worlds,
whose truth is fixed, the data and the noise; on its `drawn` worlds the channels' curves too. A
family's environments are pooled into one sample. Each arm made one plan for each world and rung,
so the Monte Carlo error of a sampled fit is inside the spread of the differences: no world was
fitted again with other seeds to separate it. The intervals say nothing about worlds outside the
families.

**Worlds per cell.** The complete cases each cell above is read on: the worlds where both the arm
and the rival made a plan, from the run's reading (`scripts/readme_record.py` writes the table).

<!-- record:cells -->
| Family | Rung | Arm | Status quo | Equal split | Meridian | PyMC-Marketing | Orbit |
|---|---|---|---|---|---|---|---|
| 0 Track M v2 | 0 | `margin held` | 350 | 350 | 350 | 350 | 350 |
| 0 Track M v2 | 4 | `gs held` | 350 | 350 | 350 | 350 | 350 |
| 2 Spend confounded with demand | 0 | `margin held` | 240 | 240 | 240 | 240 | 240 |
| 2 Spend confounded with demand | 4 | `gs held` | 240 | 240 | 240 | 240 | 240 |
| 3 Collinear channels | 0 | `margin held` | 270 | 270 | 270 | 270 | 270 |
| 3 Collinear channels | 4 | `gs held` | 270 | 270 | 270 | 270 | 270 |
| 5 Budget follows performance | 0 | `margin held` | 240 | 240 | 240 | 240 | 240 |
| 5 Budget follows performance | 4 | `gs held` | 240 | 240 | 240 | 240 | 240 |
| 7 Short series | 0 | `margin held` | 239 (the arm failed 1) | 239 (the arm failed 1) | 239 (the arm failed 1) | 239 (the arm failed 1) | 239 (the arm failed 1) |
| 7 Short series | 1–2 | `gs held` | 240 | 240 | 240 | 240 | 240 |
| 13 PyMC-Marketing's home ground | 4 | `gs held` | 240 | 240 | 240 | 240 | 240 |
| 8 Geo | 0 | `geo eb` | 240 | 240 | 240 | 235 (the rival failed 5) | 240 |
| 8 Geo | 4 | `geo eb` | 240 | 240 | 240 | 234 (the rival failed 6) | 240 |
| 9 Spillover between geos | 0 | `geo eb` | 240 | 240 | 240 | 233 (the rival failed 7) | 240 |
| 9 Spillover between geos | 4 | `geo eb` | 240 | 240 | 240 | 234 (the rival failed 6) | 240 |
| 10 Few geos | 0 | `geo eb` | 240 | 240 | 240 (exploratory) | 236 (the rival failed 4) | 240 |
| 10 Few geos | 4 | `geo eb` | 240 | 240 | 240 (exploratory) | 227 (the rival failed 13) | 240 |
<!-- /record:cells -->

What this measured, rung by rung:

- **k = 0.** `margin held` has a lower regret than every rival on every confirmatory national
  family (0, 2, 3, 5 and 7): Meridian's by 0.02 to 0.05 a euro.
- **The tested rung.** `gs held` has a lower regret than Meridian, Orbit and the status quo on 6 of
  6 families, and than the equal split on 5 (it ties on family 13). Against PyMC-Marketing it is
  level: −0.007 to +0.020, every interval below δ.
- **The geo families.** `geo eb` has a lower regret than Meridian, Orbit, the status quo and the
  equal split on families 8, 9 and 10. Against PyMC-Marketing it is level: −0.008 to +0.005.

## Where it loses

These were read in the same run. They are as much a part of the result as the table above.

- **S-curves with no lift test (family 4, k = 0): not within δ.** `margin held` loses to
  PyMC-Marketing by +0.184 [+0.136, +0.232] and to Orbit by +0.171 [+0.118, +0.223] a euro. This is
  the run's largest loss. At the tested rung, `gs held` has a lower regret than every rival on the
  same family.
- **The tested rung on drifting effects (family 1) and on truth outside the history's range
  (family 6): not within δ.** `gs held` loses to PyMC-Marketing by +0.029 [+0.002, +0.055] and
  +0.039 [+0.022, +0.055].
- **Two losses within δ in the claim.** Holm's step-down at familywise 0.025, over 88
  comparisons (the 83 claimed and family 13's five at k = 0), finds two losses, both to
  PyMC-Marketing at the tested rung: family 3, +0.008 [+0.003, +0.013], and family 5, +0.020
  [+0.010, +0.030]. They lie inside δ, so the verdicts hold, but they are losses. On family 9 at
  the tested rung `geo eb` loses to PyMC-Marketing by +0.005 [+0.002, +0.009], unadjusted. At
  δ = 0.01, 4 of the 83 claimed comparisons are not within: families 2, 3 and 5 at the tested rung
  against PyMC-Marketing, and family 13 at the tested rung against the equal split.
- **Exploratory losses within δ.** On siMMMulator's worlds (family 15), `margin held` loses to
  PyMC-Marketing at k = 0 by +0.019 [+0.010, +0.028], and `gs held` loses to Meridian,
  PyMC-Marketing and Orbit at the tested rung by +0.007 to +0.012. On the long-term family (18),
  `gs held` loses to PyMC-Marketing at the tested rung by +0.008 [+0.000, +0.017].
- **No headline.** Family 12 (Meridian's priors) at both rungs, and family 13 (PyMC-Marketing's
  priors) at k = 0, needed more worlds than the run's cap. They were run and read, but they are
  exploratory, so "within δ of every maintained MMM" is not said. Read there, every comparison
  with the table's five rivals is within δ; on family 12 at k = 0 `margin held` less Meridian is
  −0.022 [−0.044, +0.001], an interval that crosses 0 (Hall's reader reads it as `beats`).
- **Families 1, 4, 6, 14 and 15 are exploratory.** They were read on the worlds of the previous
  scored run (M2), against the rivals' records from that run, not on fresh worlds.

## The tuned rivals

Each rival ran at its recommended settings. Beside two of them ran the setting that the tool's own
criterion chose on the pilot's worlds, before any confirmatory world and never by regret
(`scorecard/tuned.json`; the pre-registration's Arms): `meridian tuned`, Meridian with
`contribution` priors and knots at most 13 weeks apart, chosen by its health card and its
out-of-sample error; `robyn tuned`, Robyn with `ts_validation` on. PyMC-Marketing's tuning chose
its recommended setting, so it has no tuned column. No development record holds a tuned setting,
so neither tuned rival has a predictive: both are read on their roots' worlds, exploratory, and
enter no verdict. Robyn ran on 20 worlds an environment of families 9, 12 and 13.

Each cell is the claim's arm less the rival, regret per euro, with its paired 95 % Student-t
interval and the reader's word, as the run's reading prints them; a Robyn cell names its worlds.
The word is Hall's reader's, which corrects for the differences' skew: on a skewed sample it can
read `beats` where the Student interval crosses 0.

<!-- record:tuned -->
| Family | Rung | Arm | Meridian | Meridian tuned | Robyn | Robyn tuned |
|---|---|---|---|---|---|---|
| 0 Track M v2 | 0 | `margin held` | −0.0515 [−0.0722, −0.0309], beats | −0.1206 [−0.1392, −0.1019], beats | not run | not run |
| 0 Track M v2 | 4 | `gs held` | −0.0764 [−0.0853, −0.0676], beats | −0.0737 [−0.0839, −0.0635], beats | not run | not run |
| 2 Spend confounded with demand | 0 | `margin held` | −0.0437 [−0.0688, −0.0186], beats | −0.0700 [−0.0927, −0.0473], beats | not run | not run |
| 2 Spend confounded with demand | 4 | `gs held` | −0.0680 [−0.0817, −0.0542], beats | −0.0840 [−0.1006, −0.0673], beats | not run | not run |
| 3 Collinear channels | 0 | `margin held` | −0.0211 [−0.0402, −0.0020], beats | −0.0420 [−0.0622, −0.0218], beats | not run | not run |
| 3 Collinear channels | 4 | `gs held` | −0.0816 [−0.0934, −0.0697], beats | −0.0936 [−0.1081, −0.0792], beats | not run | not run |
| 5 Budget follows performance | 0 | `margin held` | −0.0421 [−0.0617, −0.0225], beats | −0.0582 [−0.0781, −0.0383], beats | not run | not run |
| 5 Budget follows performance | 4 | `gs held` | −0.0690 [−0.0834, −0.0545], beats | −0.0820 [−0.0988, −0.0653], beats | not run | not run |
| 7 Short series | 0 | `margin held` | −0.0453 [−0.0664, −0.0242], beats | −0.0528 [−0.0714, −0.0342], beats | not run | not run |
| 7 Short series | 1–2 | `gs held` | −0.0616 [−0.0776, −0.0457], beats | −0.0739 [−0.0923, −0.0555], beats | not run | not run |
| 8 Geo | 0 | `geo eb` | −0.0429 [−0.0463, −0.0394], beats | −0.0556 [−0.0603, −0.0509], beats | not run | not run |
| 8 Geo | 4 | `geo eb` | −0.0443 [−0.0477, −0.0409], beats | −0.0554 [−0.0600, −0.0508], beats | not run | not run |
| 9 Spillover between geos | 0 | `geo eb` | −0.0479 [−0.0543, −0.0416], beats | −0.0665 [−0.0739, −0.0590], beats | −0.0356 [−0.0527, −0.0186], beats, 40 worlds | −0.0401 [−0.0591, −0.0211], beats, 40 worlds |
| 9 Spillover between geos | 4 | `geo eb` | −0.0442 [−0.0505, −0.0379], beats | −0.0687 [−0.0779, −0.0595], beats | −0.0741 [−0.0980, −0.0502], beats, 40 worlds | −0.0873 [−0.1177, −0.0570], beats, 40 worlds |
| 10 Few geos | 0 | `geo eb` | −0.0307 [−0.0355, −0.0259], beats | −0.0335 [−0.0387, −0.0284], beats | not run | not run |
| 10 Few geos | 4 | `geo eb` | −0.0339 [−0.0389, −0.0289], beats | −0.0370 [−0.0423, −0.0317], beats | not run | not run |
| 12 Meridian's home ground | 0 | `margin held` | −0.0216 [−0.0445, +0.0013], beats | −0.0338 [−0.0599, −0.0078], beats | −0.1368 [−0.3813, +0.1078], ties, 20 worlds | −0.2104 [−0.4753, +0.0545], beats, 20 worlds |
| 12 Meridian's home ground | 4 | `gs held` | −0.0128 [−0.0226, −0.0031], beats | −0.0358 [−0.0536, −0.0180], beats | −0.3061 [−0.5726, −0.0396], beats, 20 worlds | −0.3341 [−0.6172, −0.0509], beats, 20 worlds |
| 13 PyMC-Marketing's home ground | 0 | `margin held` | −0.0102 [−0.0299, +0.0094], ties | −0.0618 [−0.0823, −0.0413], beats | −0.1850 [−0.2764, −0.0937], beats, 20 worlds | −0.2928 [−0.4187, −0.1670], beats, 20 worlds |
| 13 PyMC-Marketing's home ground | 4 | `gs held` | −0.0250 [−0.0458, −0.0043], beats | −0.0916 [−0.1162, −0.0670], beats | −0.1025 [−0.2471, +0.0421], ties, 20 worlds | −0.1686 [−0.3428, +0.0055], beats, 20 worlds |
| 17 Twins | 0 | `margin held` | −0.4751 [−0.5188, −0.4313], beats | −0.6714 [−0.6808, −0.6619], beats | not run | not run |
| 17 Twins | 4 | `gs held` | −0.0224 [−0.0300, −0.0148], beats | −0.0634 [−0.0730, −0.0537], beats | not run | not run |
| 18 Long term | 0 | `margin held` | −0.0348 [−0.0606, −0.0089], beats | −0.0604 [−0.0856, −0.0352], beats | not run | not run |
| 18 Long term | 4 | `gs held` | −0.0730 [−0.0914, −0.0547], beats | −0.0874 [−0.1097, −0.0651], beats | not run | not run |
| 19 Flat | 0 | `margin held` | not compared: the rival failed 180 | not compared: the rival failed 180 | not run | not run |
| 19 Flat | 4 | `gs held` | not compared: the rival failed 180 | not compared: the rival failed 180 | not run | not run |
| 20 Promotions | 0 | `margin held` | −0.0246 [−0.0455, −0.0036], beats | −0.0434 [−0.0768, −0.0100], beats | not run | not run |
| 20 Promotions | 4 | `gs held` | −0.0842 [−0.1007, −0.0676], beats | −0.0656 [−0.0806, −0.0505], beats | not run | not run |

Over these rows the claim's arm reads: against Meridian it beats on 25, ties on 1, is not compared
on 2, of 28; against Meridian tuned it beats on 26, is not compared on 2, of 28; against Robyn it
beats on 4, ties on 2, of 6; against Robyn tuned it beats on 6, of 6.
<!-- /record:tuned -->

## What the run did not cover

- **Bench families not scored.** Family 11 (Google's AMSS simulator) was withdrawn. Family 16, a
  published third-party instance of endogenous spend chosen in its place, was left out: its files
  and their licence had not arrived when the run was frozen. Family 21 (mmm-framework's own
  worlds) was not in the run: its truth states each channel's contribution and return and no
  response curve, so it scores a model's reading, not a plan's regret. Family 10's 50-geo
  environment was not run: its only reading is an audit on fresh worlds before the run. Family 1's
  `cpm` environment and family 2's `search-demand` environment were not in the run's
  declaration.
- **Families 17 to 20** (twins one history cannot tell apart, a long-term brand measure, a flat
  channel, promotions) are exploratory. On family 17 the time guard plays the status quo on every
  world, by design. Orbit could not be compared on family 17 in the run: the bench's Orbit script
  planned exactly three channels and those worlds have two, a harness limit, not Orbit's. After the
  run the bench fixed its script (branch `orbit-two-channels`, commit `174228b`) and Orbit was
  re-run on those worlds: `margin held` less Orbit read −0.169 [−0.208, −0.131] at k = 0 and
  `gs held` less Orbit −0.082 [−0.106, −0.057] at the tested rung. That reading is post hoc and
  enters no verdict.
- **Robyn 3.12.1** ran on 20 worlds an environment of families 9, 12 and 13 only, exploratory, in
  no verdict. The tuned rivals (`meridian tuned`, `robyn tuned`) are read beside the claim, never
  in it ("The tuned rivals", above).
- **PyMC-Marketing ran its static model on every family**: logistic saturation and a static
  intercept, as pre-registered, with no time-varying term. Its results on the drift family (1) are
  that model's.
- **The claim's arms were designed after the run's pilot**, on the development set (M2's scored
  worlds) and on audit worlds, as the pre-registration records (its Deviations, items 1 to 3).
  Their predictive of the verdicts could be optimistic; on the fresh worlds 17 of 17 held, against
  16.94 expected.
- **Simulated worlds only.** No claim here is about real data (below).

## Pre-registration and reproduction

- The run is M3. Its pre-registration is `scorecard/preregistration/m3.md`, with `m3.json` and
  `m3_sizing.py` beside it. It was frozen on 2026-11-17, before any confirmatory world was fitted,
  in this project's then-private repository. Its SHA-256 is
  `d684b25ccae0e3574fe2f962097e399c9f400f0641e52431cf794e131763d924`, which
  `sha256sum scorecard/preregistration/m3.md` prints.
- **There is no public time-stamp before the run.** The repository was private when the file was
  frozen and while the run ran (2026-11-17 to 2026-11-20). Nothing public shows that the file was
  not changed after the results were known; the hash shows only that this file is the one the
  run declared.
- This repository starts from a fresh history at 0.1.0. The engine and the scorecard's harness in
  its first commit are the code the run ran: their git tree hashes,
  `782ca30c324a5720c37c4b4e6856862d4f22a2f5` for `src/` and
  `32c5f8714d670141efc9f9c636fce410975ad2bb` for `scorecard/`, are those of the commit the run
  declared (`git rev-parse v0.1.0:src v0.1.0:scorecard`). The run locked causal-hybrid-control
  1.3.0; 0.1.0 locks 1.3.1, whose changes the engine does not call.
- The bench is public: [causaldyn/causaldyn-bench](https://github.com/causaldyn/causaldyn-bench),
  commit `67f92b7478b0a1423538d77e62d8198923c52a85` (branch `m3`), each rival's environment pinned
  by its lock file under `scripts/`.
- The harness is `scorecard/`. `python -m scorecard.run` declares a run, exports its worlds, runs
  every arm and scores every record (its docstring gives the environment); `python -m
  scorecard.claim read --recorded <roots>` reads the claim, the verdicts, Holm's losses and every
  recorded arm less every rival. The run took 5 201 CPU-hours (Compute, below). Its per-world
  records are not published yet, so the reading cannot be re-run without re-running the worlds.
- **The tables in the sections on worlds per cell, the tuned rivals, versions and compute are
  written, not typed.** `scripts/readme_record.py extract` read them from the run's records and
  its reading into `evidence/results/m3_record.json`, with the SHA-256 of every file they came
  from; `python scripts/readme_record.py` writes them here from that file, and
  `tests/test_readme.py` fails where a table differs from what it writes.

## Versions of every arm

What each arm's records state: the packages and the Python each record names, the commits of the
engine and the bench where the record names them, and the settings each record holds. A setting
marked "by family" took each value on some families: a history's length sets a kernel's or a
lag's weeks. The rest of each arm's setting is in the pre-registration's Arms table. The rivals'
records reused from M2 name the same versions.

<!-- record:versions -->
| Arm | Role | Version, as each record states it | Run with | How it was configured |
|---|---|---|---|---|
| `margin held` | the claim's arm at k = 0; families 0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15, 17, 18, 19, 20 | bench at `67f92b7`, mixdyn at `183ca75` | Python 3.14.10; causal-hybrid-control 1.3.0, jax 0.12.0, numpy 2.5.4, scipy 1.18.1 | planner water-filling; retentions 0.1, 0.3, 0.5, 0.7, 0.9; kernel weeks 6, 9, 156 or 1 (by family); AR(1) grid 0.3, 0.6, 0.8, 0.9, 0.95, 0.98; time guard 0.8 |
| `anchored held` | the claim's arm at the tested rung, `gs held`: its router; families 0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15, 17, 18, 19, 20 | bench at `67f92b7`, mixdyn at `183ca75` | Python 3.14.10; causal-hybrid-control 1.3.0, jax 0.12.0, numpy 2.5.4, scipy 1.18.1 | planner water-filling on a table; curves michaelis-menten, logistic; lift rows read 12, 3, 6 or 8 (by family); routes a flagged world to stacked; time guard 0.8 |
| `stacked` | `gs held`'s routed plan, the Bayesian engine's; families 0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 20 | bench at `67f92b7`, mixdyn at `183ca75` | Python 3.14.10; causal-hybrid-control 1.3.0, jax 0.12.0, numpy 2.5.4, scipy 1.18.1 | sampler chains 4, draws 1000, warmup 1000; curves hill, tanh; planner plan_posterior |
| `geo eb` | the claim's arm on the geo families; families 8, 9, 10 | bench at `67f92b7`, mixdyn at `183ca75` | Python 3.14.10; causal-hybrid-control 1.3.0, jax 0.12.0, numpy 2.5.4, scipy 1.18.1 | planner plan_mixture; retentions 0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9; curvatures 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, inf; sweeps 3 |
| `chc`, `decisions`, `dlm`, `geo-dlm`, `read-sc`, `read-sdid`, `read-tbr`, `ridge`, `switchback`, `voi` | the engine's other arms, recorded beside the claim; families 0, 2, 3, 5, 7, 8, 9, 10, 12, 13, 17, 18, 19, 20 | bench at `67f92b7`, mixdyn at `183ca75` | Python 3.14.10; causal-hybrid-control 1.3.0, jax 0.12.0, numpy 2.5.4, scipy 1.18.1 | each record's own settings |
| Meridian | rival; families 0, 2, 3, 5, 7, 8, 9, 10, 12, 13, 17, 18, 20; M2's records reused on 1, 4, 6, 14, 15; 360 failure records, family 19 | Meridian 2.1.0 | Python 3.12.16; arviz 0.19.0, jax 0.11.2, jaxlib 0.11.2, numpy 2.3.5, pandas 2.3.3, tensorflow 2.21.0, tfp-nightly 0.26.0.dev20260130, xarray 2026.9.0 | sampling n_adapt 1000, n_burnin 500, n_chains 10, n_keep 1000; automatic knot selection on; ROI prior at k = 0: loc 0.2; scale 0.9; max lag 5, 7, 8 or 0 (by family); dtype float64 |
| Meridian tuned | tuned rival; families 0, 2, 3, 5, 7, 8, 9, 10, 12, 13, 17, 18, 20; 360 failure records, family 19 | Meridian 2.1.0 | Python 3.12.16; arviz 0.19.0, jax 0.11.2, jaxlib 0.11.2, numpy 2.3.5, pandas 2.3.3, tensorflow 2.21.0, tfp-nightly 0.26.0.dev20260130, xarray 2026.9.0 | sampling n_adapt 1000, n_burnin 500, n_chains 10, n_keep 1000; automatic knot selection off; media prior contribution; knots, weeks apart at most 13; max lag 5, 7, 8 or 0 (by family); dtype float64 |
| PyMC-Marketing | rival; families 0, 2, 3, 5, 7, 8, 9, 10, 12, 13, 17, 18, 19, 20; M2's records reused on 1, 4, 6, 14, 15; 49 failure records, families 8, 9, 10, 15 | PyMC-Marketing 1.3.0 | Python 3.12.16; arviz 1.3.0, numpy 2.4.6, pandas 3.1.0, pymc 6.3.2, pytensor 3.3.6, scipy 1.18.1 | model static; chains 4; draws 1000; adstock weeks 6, 8, 9 or 1 (by family); yearly harmonics 3 |
| Orbit | rival; families 0, 2, 3, 5, 7, 8, 9, 10, 12, 13, 18, 19, 20; M2's records reused on 1, 4, 6, 14, 15; 242 failure records, families 17, 18 | Orbit 1.1.5.1 | Python 3.12.16; cmdstanpy 1.3.0, numpy 2.5.4, pandas 3.0.6, pyro-ppl 1.9.2, scipy 1.18.1, torch 2.14.1+cpu | model log-log KTR (Ng, Wang and Dai 2021); samples 400; steps KTR's default; holdout weeks 13; seasonality 52 |
| Robyn | rival, read beside the claim; families 9, 12, 13 | Robyn 3.12.1 | R version 4.6.2; StanHeaders 2.39.1, doParallel 1.0.17, doRNG 1.8.6.3, dplyr 1.2.1, foreach 1.5.2, glmnet 5.1, jsonlite 2.0.0, nloptr 2.2.1, prophet 1.1.7, reticulate 1.47.0, rstan 2.32.7; Python 3.13.18, nevergrad 1.0.12, numpy 2.5.3 | ranges genre; iterations 2000; trials 5 or 10 (by family); cores 4; date range last_13; calibration total |
| Robyn tuned | tuned rival, read beside the claim; families 9, 12, 13 | Robyn 3.12.1 | R version 4.6.2; StanHeaders 2.39.1, doParallel 1.0.17, doRNG 1.8.6.3, dplyr 1.2.1, foreach 1.5.2, glmnet 5.1, jsonlite 2.0.0, nloptr 2.2.1, prophet 1.1.7, reticulate 1.47.0, rstan 2.32.7; Python 3.13.18, nevergrad 1.0.12, numpy 2.5.3 | ranges genre; iterations 2000; trials 5 or 10 (by family); cores 4; date range last_13; calibration total; ts_validation on |
<!-- /record:versions -->

## Compute

Each process the run started is one line of its root's `logs/costs.jsonl`: one arm on one world,
at the rungs it planned. Its core-hours are its scope's CPU time, its peak memory its scope's
`memory.peak` (the memory charged to the scope, page cache included: not the same measure as a peak
RSS), and its wall the time from its start to its end. A failed attempt that ended by itself is a
line too; a process stopped at its timeout or by one of the box's spot interruptions wrote none, so
its CPU time is not counted here. The CPU model and the threads (`OMP_NUM_THREADS`) are what each
record's `cost` names; the CPU quota is the CPUs the process was given, its cost class's in the
stage's plan. The processes shared one m6a.48xlarge (the pre-registration's Compute): the figures
say what was spent under that load, not how fast a tool is.

<!-- record:compute -->
One box of 192 CPUs and 740 GiB ran every process, admitting up to 188 CPUs of quota at a time. It
was launched 4 times: the stage resumed after each of 3 spot interruptions.

| Family | CPU model | Threads | Processes | Peak memory a process, median / most | Wall a process, median / most | Core-hours: engine | rivals | all |
|---|---|---|---|---|---|---|---|---|
| 0 Track M v2 | AMD EPYC 7R13 Processor | 1 | 5 489 | 0.87 / 9.15 GiB | 138 s / 33 min | 315.8 | 139.2 | 455.0 |
| 1 Drift | AMD EPYC 7R13 Processor | 1 | 720 | 0.17 / 0.89 GiB | 14 s / 10 min | 35.4 | 0.0 | 35.4 |
| 2 Spend confounded with demand | AMD EPYC 7R13 Processor | 1 | 3 600 | 0.85 / 10.67 GiB | 141 s / 39 min | 196.2 | 100.9 | 297.1 |
| 3 Collinear channels | AMD EPYC 7R13 Processor | 1 | 4 050 | 0.85 / 8.61 GiB | 140 s / 48 min | 229.5 | 111.2 | 340.7 |
| 4 S-curves | AMD EPYC 7R13 Processor | 1 | 720 | 0.17 / 0.89 GiB | 14 s / 12 min | 33.9 | 0.0 | 33.9 |
| 5 Budget follows performance | AMD EPYC 7R13 Processor | 1 | 3 600 | 0.85 / 7.66 GiB | 141 s / 37 min | 193.2 | 99.1 | 292.3 |
| 6 Truth outside the observed range | AMD EPYC 7R13 Processor | 1 | 720 | 0.17 / 0.91 GiB | 14 s / 11 min | 34.3 | 0.0 | 34.3 |
| 7 Short series | AMD EPYC 7R13 Processor | 1 | 3 600 | 0.71 / 9.67 GiB | 129 s / 25 min | 129.5 | 86.0 | 215.4 |
| 8 Geo | AMD EPYC 7R13 Processor | 1 | 3 230 | 0.69 / 20.23 GiB | 333 s / 42 min | 174.2 | 456.6 | 630.7 |
| 9 Spillover between geos | AMD EPYC 7R13 Processor | 1 | 3 082 | 1.23 / 20.49 GiB | 324 s / 72 min | 10.5 | 880.8 | 891.4 |
| 10 Few geos | AMD EPYC 7R13 Processor | 1 | 2 880 | 0.74 / 19.78 GiB | 130 s / 49 min | 9.5 | 296.9 | 306.4 |
| 12 Meridian's home ground | AMD EPYC 7R13 Processor | 1 | 3 701 | 0.86 / 11.60 GiB | 132 s / 72 min | 176.7 | 286.6 | 463.3 |
| 13 PyMC-Marketing's home ground | AMD EPYC 7R13 Processor | 1 | 3 706 | 0.87 / 9.30 GiB | 131 s / 72 min | 175.3 | 291.2 | 466.5 |
| 14 Robyn's home ground | AMD EPYC 7R13 Processor | 1 | 720 | 0.17 / 0.89 GiB | 13 s / 10 min | 32.8 | 0.0 | 32.8 |
| 15 siMMMulator 1.1.0 | AMD EPYC 7R13 Processor | 1 | 300 | 0.17 / 0.95 GiB | 13 s / 24 min | 14.9 | 0.0 | 14.9 |
| 17 Twins | AMD EPYC 7R13 Processor | 1 | 1 800 | 0.72 / 2.29 GiB | 112 s / 16 min | 62.8 | 39.0 | 101.8 |
| 18 Long term | AMD EPYC 7R13 Processor | 1 | 1 800 | 0.88 / 9.60 GiB | 159 s / 40 min | 108.2 | 54.8 | 163.0 |
| 19 Flat | AMD EPYC 7R13 Processor | 1 | 2 700 | 0.52 / 5.86 GiB | 88 s / 33 min | 148.9 | 39.6 | 188.4 |
| 20 Promotions | AMD EPYC 7R13 Processor | 1 | 2 700 | 0.85 / 6.66 GiB | 150 s / 39 min | 159.1 | 78.9 | 238.0 |
| all | | | 49 118 | | | 2 240.6 | 2 960.8 | 5 201.3 |

| Arm | Role | CPU quota | Rungs a process | Families | Processes | Peak memory a process, median / most | Wall a process, median / most | Core-hours |
|---|---|---|---|---|---|---|---|---|
| `dlm` | engine | 1 | 1 | 0, 2, 3, 5, 7, 12, 13, 17, 18, 19, 20 | 4 839 | 3.55 / 6.80 GiB | 18 min / 48 min | 1 424.8 |
| `meridian` | rival | 1.2 | 1 | 0, 2, 3, 5, 7, 8, 9, 10, 12, 13, 17, 18, 19, 20 | 6 279 | 3.09 / 20.49 GiB | 218 s / 46 min | 786.6 |
| `meridian tuned` | rival | 1.2 | 1 | 0, 2, 3, 5, 7, 8, 9, 10, 12, 13, 17, 18, 19, 20 | 6 278 | 2.05 / 20.47 GiB | 182 s / 43 min | 692.4 |
| `pymc-marketing` | rival | 1.5 | 1 | 0, 2, 3, 5, 7, 8, 9, 10, 12, 13, 17, 18, 19, 20 | 6 259 | 0.75 / 1.25 GiB | 139 s / 49 min | 519.4 |
| `bayes` | engine | 1 | 1 | 0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 20 | 3 600 | 0.85 / 0.95 GiB | 450 s / 38 min | 502.1 |
| `robyn` | rival | 3 | 1 | 9, 12, 13 | 172 | 1.32 / 2.03 GiB | 39 min / 72 min | 401.9 |
| `robyn tuned` | rival | 3 | 1 | 9, 12, 13 | 169 | 1.37 / 2.02 GiB | 30 min / 72 min | 401.4 |
| `orbit` | rival | 1 | 1 | 0, 2, 3, 5, 7, 8, 9, 10, 12, 13, 17, 18, 19, 20 | 6 282 | 0.41 / 0.83 GiB | 95 s / 143 s | 159.2 |
| `chc` | engine | 1 | 2 | 0, 2, 3, 5, 7, 8, 9, 10, 12, 13, 17, 18, 19, 20 | 3 220 | 0.88 / 0.94 GiB | 89 s / 153 s | 78.2 |
| `switchback` | engine | 1 | 1 | 8 | 60 | 0.72 / 0.76 GiB | 36 min / 42 min | 35.5 |
| `ridge` | engine | 1 | 2 | 0, 2, 3, 5, 7, 8, 9, 10, 12, 13, 17, 18, 19, 20 | 3 140 | 0.25 / 0.26 GiB | 36 s / 118 s | 32.7 |
| `read-sdid` | engine | 1 | 1 | 8 | 60 | 0.69 / 0.73 GiB | 33 min / 37 min | 31.7 |
| `read-sc` | engine | 1 | 1 | 8 | 60 | 0.69 / 0.73 GiB | 32 min / 37 min | 31.6 |
| `read-tbr` | engine | 1 | 1 | 8 | 60 | 0.69 / 0.73 GiB | 32 min / 37 min | 31.6 |
| `voi` | engine | 1 | 1 | 0 | 120 | 1.02 / 1.11 GiB | 14 min / 17 min | 30.2 |
| `decisions` | engine | 1 | 1 | 0 | 60 | 0.88 / 0.94 GiB | 12 min / 14 min | 12.5 |
| `margin` | engine | 1 | 1 or 2 | 0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15, 17, 18, 19, 20 | 3 480 | 0.17 / 0.18 GiB | 13 s / 17 s | 12.4 |
| `anchored` | engine | 1 | 1 | 0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15, 17, 18, 19, 20 | 3 480 | 0.17 / 0.18 GiB | 12 s / 15 s | 10.9 |
| `geo-eb` | engine | 1 | 1 | 8, 9, 10 | 1 440 | 0.17 / 0.18 GiB | 11 s / 15 s | 4.4 |
| `geo-dlm` | engine | 1 | 1 | 8 | 60 | 0.30 / 0.30 GiB | 128 s / 170 s | 2.1 |

Families 1, 4, 6, 14, 15 read the rivals' records of M2, whose compute is not counted here; those
records name AMD EPYC 7R13 Processor. The engine's core-hours include the arms recorded beside the
claim, read by no comparison: 50.9 on family 0, 164.7 on family 8.
<!-- /record:compute -->

## Real-data validation: none, and why

No public panel carries the truth an engine would be scored against: a real market's returns are
never observed, only estimated. Every claim about the engine is therefore measured on simulated
worlds whose truth is known, the bench's families, and no claim here is about real data.

What real data can score is a prediction of a lift or geo test the model did not read: a test's
reading is a noisy but unbiased measurement of a quantity every engine predicts.
`scripts/real_data_protocol.py` is the protocol a partner would run on one weekly panel with at
least two such tests. Each test is withheld in turn; every engine fits the panel and the other
tests and predicts the withheld test's lift; the cells are `|predicted - test| / SE`, each test's
standard error, and the share of tests whose 95 % interval holds the prediction. It runs end to
end on a synthetic stand-in (`python scripts/real_data_protocol.py`), and
`tests/test_real_data_protocol.py` runs it on a small one. No partner has run it yet, so no real
panel's cells exist.

## Using it

`SKILL.md` walks a person, or an agent, through one panel: read, fit, diagnostics, tests, plan and
report, each step gated by the engine's own checks, because a model that fits is not one whose
returns are right. `docs/api/index.md` lists the public names and the versioning rules,
`docs/scope.md` what the engine does not model (reach and frequency among it), and `evidence/` the
scripts and records the ADRs' measurements come from.

Each channel's return can be read against an unmeasured confounder: `mixdyn.inference.sensitivity`
gives Cinelli and Hazlett's robustness values for the ridge engine and the marginal sensitivity
model's `Gamma*` for the Bayesian ones, `mixdyn.api.sensitivity_line` states them, and
`mixdyn.api.factsheet` writes each Bayesian channel's line among its limitations (ADR 0061). The
bound is wide wherever the spend moved little beside the KPI's unexplained variation, confounded
or not.

A static fit can be checked on the periods after its window and carried forward as weeks arrive
(ADR 0079). `holdout_bayes` scores the posterior predictive on the window's last periods, held out
of the fit: periods drawn from the whole window would share their autocorrelated noise with their
fitted neighbours and flatter the score. `refresh_bayes` refits the whole panel with each
channel's return prior tempered by the previous posterior, and reports how far each return moved
and whether it left the previous 80 % interval. The scorecard's refresh cell
(`python -m scorecard.refresh`) refits every arm on a world's history four weeks at a time, six
times, and reads how far each return and each plan's regret swing between refits, and, from a
rolling origin on the short-history family, each origin's forecast score. Measurement services
sell a refresh loop whose results stay stable; the cell measures that stability on worlds whose
truth is known, for this engine's arms and the rivals'.

<!-- --8<-- [end:record] -->
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
