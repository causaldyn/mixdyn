# mixdyn

A media-mix modelling (MMM) engine in Python and JAX: regressions, discount dynamic linear models,
Bayesian models with lift and geo tests in the likelihood, geo hierarchies, test readers and budget
plans, computed without a UI. Built on
[causal-hybrid-control](https://github.com/causaldyn/causal-hybrid-control). Apache-2.0.

What it has been measured to do, and where it loses, is below: one pre-registered run of the
[causaldyn bench](https://github.com/causaldyn/causaldyn-bench), stated with its losses. Nothing
here says "best".

## Install

```bash
pip install mixdyn        # Python 3.12 or later
```

The runtime packages are NumPy, SciPy, JAX and causal-hybrid-control, with what those bring in: no
probabilistic-programming language and no other MMM package (ADR 0001).

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
families 8, 9, 10 and 15. On family 10, Meridian ran on a capped number of worlds: it is read
beside that verdict, not in it.

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
  with the table's five rivals is within δ; on family 12 at k = 0 `margin held` ties Meridian,
  −0.022 [−0.044, +0.001].
- **Families 1, 4, 6, 14 and 15 are exploratory.** They were read on the worlds of the previous
  scored run (M2), against the rivals' records from that run, not on fresh worlds.

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
  in it.
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
  recorded arm less every rival. The run took 5 201 CPU-hours. Its per-world records are not
  published yet, so the reading cannot be re-run without re-running the worlds.

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
