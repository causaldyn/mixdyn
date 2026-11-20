# The drift run: pre-registration

This file, `drift.json` beside it and `scorecard/drift.py` are committed before any scored world
of the run is fitted. The commit that adds them is the run's pre-registration. A change after it
is a deviation: the result names it and says why.

## Question

Where media effects drift, does the engine's dynamic linear model (DLM, ADR 0020) plan the quarter
with less regret than the static arms and the time-varying arms?

## Worlds

- **The family.** The bench's family 1, drift (`causaldyn_bench.scorecard.drift`). Each world is a
  Track M v2 drawn world whose channel effects follow one of six laws: `walk`, `trend`, `step`,
  `season`, `revert` and `static`. Under `static` the world is Track M v2's, bit for bit.
- **Scored worlds.** The first 100 of each law's scored block, seeds 221 000–221 099 (`walk`),
  222 000–222 099 (`trend`), 223 000–223 099 (`step`), 224 000–224 099 (`season`), 225 000–225 099
  (`revert`) and 226 000–226 099 (`static`), at rungs k = 0 and k = 4 of the lift-test ladder:
  1 200 world-rungs.
- **Why 100.** The sample-size rule (`scorecard.drift.worlds`): for each law and each arm the
  primary is paired with at k = 0, the fewest worlds, in tens from 30 to 100, at which the pilot's
  spread gives a 95 % half-width of at most half the pilot's difference; the most over every law
  and arm. Every law reaches the ceiling: each has an arm the pilot cannot tell from the primary at
  fewer than 100.
- **The pilot.** The first 10 worlds of each law's pilot block (220 000–220 009, 220 100–220 109,
  …, 220 500–220 509) at both rungs, 120 world-rungs. The tools' records and the DLM's are the
  pilot's; the static arm was fitted and planned again at mixdyn `7dd4c57` (ADR 0036). Its
  `scores.jsonl` has SHA-256 `86d5c29baab6c95d8ca03198d589d24287fd5c49d26f7923010ce81dfe5ee103`
  and is archived with the scored run's records. P7 reads the PyMC-Marketing arm fitted again at
  bench `1f4161c` on the pilot's drifting worlds at k = 0, its `multiplier` blocks set into the
  pilot's records of those arms; the pilot scored so has SHA-256
  `29c6a7d0f8ad19e053006beb9f49376318565db148be425ad391c602c680ece4`, and `drift.json` is made
  from it. No pilot world is a scored world.

## Arms

| Arm | What it plans on | Pin and setting |
|---|---|---|
| `dlm mixture` | the DLM's kept specs, the posterior mixture's expected gain | this commit |
| `dlm plug-in`, `dlm cvar 0.1` | the same fit: the best spec's plug-in; the worst tenth's mean gain | this commit |
| `stacked`, `cvar 0.1` | the static Bayesian model, two curve families stacked: the expected gain; the worst tenth's mean gain | this commit |
| `pymc-marketing` | PyMC-Marketing 1.2.0's MMM, static | the bench's `scripts/pymc_marketing_arm.py`, `--setting static` |
| `pymc-marketing shared` | the same, one HSGP multiplier shared by the channels | `--setting shared` |
| `pymc-marketing channel` | the same, a `SoftPlusHSGP` for each channel | `--setting channel` |
| `meridian` | Meridian 2.1.0 | the bench's `scripts/meridian_arm.py` |
| `orbit` | Orbit KTR 1.1.5.1, log-log | the bench's `scripts/orbit_arm.py` |
| `robyn` | Robyn 3.12.1, genre ranges, 5 trials | the bench's `scripts/robyn_arm.R`; 50 worlds only, below |
| `status quo`, `equal split` | no model | built in |

- Every tool runs its documented setting in its pinned environment (the bench's `scripts/*_arm`
  lock files) and is told the kernel's length, Robyn excepted, which cannot be.
- **Robyn** costs 1.9–3.9 core-hours a world, so it plans only the first 10 scored worlds of each
  drifting law at k = 0, 50 worlds. It runs as a second run declared on those worlds, every other
  record hard-linked from the main run, and its comparison reads those 50 worlds alone. The pilot
  did not run it, so it carries no prediction.
- **The engine's default procedure** is `dlm mixture`, by the rule "X unless the pilot's paired
  interval of X less Y lies above 0" over its variants, pooled over the drifting laws at k = 0:
  `dlm mixture` less `dlm plug-in` +0.0126 [−0.0050, +0.0302], less `dlm cvar 0.1` −0.0401
  [−0.1645, +0.0843]. The run's claim reads it alone.
- **PyMC-Marketing's time-varying model** the claim reads is `pymc-marketing shared`, the setting of
  least pooled regret on the pilot, a choice in its favour: 0.3778 per euro against `channel`'s
  0.4365 (static 0.4464). Both are scored and reported.

## Pins

- mixdyn at this commit. Its code is `7dd4c57`'s: the commit adds this file, `drift.json`,
  `scorecard/drift.py`, its tests and a consequence in ADR 0036.
- causal-hybrid-control 0.13.0, as mixdyn's lock holds it.
- The bench at `1f4161c`, and each tool's environment as its lock file there pins it. The pilot
  ran at `aca4eda`; at `1f4161c` every pilot export was rebuilt the same byte for byte, and the
  pilot's records scored the same. Its PyMC-Marketing arm records the multiplier (P7) and
  compiles its model without numba's file cache. With the cache, the compiled code, and so the
  draws, depend on what the cache held: one pilot world fitted from one seed drew three different
  chains from three states of it. Without it a fit is reproduced from its seed. The pilot's
  PyMC-Marketing fits ran with the cache and stand as they were drawn.
- **Every other arm is reproduced from its seed.** Each fitted pilot world 220001 (walk, k = 0)
  again: Orbit and Meridian twice each at `1f4161c`, the static arm and the DLM once at this
  commit's code. Each record equals the pilot's or the freeze's in every value but its timings and
  the path and commit of the tree it ran from, and Meridian's posterior predictive p-value, which
  its model reviewer draws with no seed (0.9888 to 0.9998 here) and which no score reads. Robyn is
  fitted twice on its first world before its run, and the result is reported with it.
- `JAX_ENABLE_X64=1`, and one thread a process (`scorecard.provenance.ONE_THREAD`).
- A scored run's arms do not start unless the bench is at its declared commit and both trees are
  clean (`scorecard.provenance.require`).

## Axes and units

- **Primary: regret per euro** of the quarter's budget: what the best plan returns over the plan's
  return, on the effect path the best plan is made for, in revenue per euro, not profit. Lower is
  better. The best plan is made on the law's expected path from the true state at the history's
  end.
- **Secondary** (`causaldyn_bench.scorecard.scoring`): the regret on the realised path; the claimed
  gain less the realised one, per euro, and whether the claim's 90 % interval holds the realised
  gain; the CRPS of the quarter's weekly sales at the status quo, in units of the history's mean
  weekly sales; each channel's window ROI and mROI, read as the share of channels whose truth the
  draws' central 90 % holds; failures; plans moved into the box; fits past their chains' bounds.

## Reading

- **The primary comparison.** At k = 0, pooled over the five drifting laws, 500 worlds:
  `dlm mixture` less each arm, paired by world, with its 95 % Student-t interval. The DLM beats the
  arm where the interval lies below 0, loses to it where the interval lies above 0, and ties it
  otherwise.
- **The claim** "the DLM beats each static arm" holds only where every interval of P1 lies below 0.
  That is an intersection–union reading, each comparison at 0.025 one-sided, and needs no
  adjustment. Every other verdict is read alone, unadjusted, and is named as such.
- **No interim look.** Every scored world is fitted and scored before any verdict is read. The run
  has no gate.
- **A failed fit** plays the status quo and is counted. **A plan off the box or the budget** is
  moved into the box and counted.
- **A fit past its chains' bounds**, an R-hat above 1.01 or more than 27 of 4 000 draws divergent as
  a share of its draws (`scorecard.report.past`): a claim is read on the plan, and the report gives
  a second reading with every such plan at the status quo's regret (P10).
- **A world where an item's quantity is not scored**, as a failed fit's CRPS, is left out of that
  item, and the reading counts it.
- **The record.** `scorecard.report` gives every law's table, rung by rung, and the pooled one;
  `python -m scorecard.drift read` reads P1–P12 against `drift.json`.

## Predictions

**The rule** (`scorecard.drift`). A verdict's prediction is the verdict most probable given the
pilot, with that chance. With the pilot's mean m, standard deviation s and n worlds, and the N
worlds the scored run reads, the scored mean is m + s (1/n + 1/N)^(1/2) T, with T Student's t on
n − 1 degrees of freedom: the predictive of a future mean under the prior uniform on the mean and on
the log of the spread. The scored interval's half-width is taken as t(0.975, N − 1) s / N^(1/2),
the pilot's spread. A share's prediction is the central 90 % of the scored count's beta-binomial
predictive under Jeffreys' prior on the pilot's count, with the chance it holds.

The predictions are forecasts, not tests. Of the 109 predictions below, 87.4 are expected to hold,
the sum of their chances; the reading reports how many did. That count checks the forecasts, not the
engine.

In the tables, "below", "across" and "above" are where the scored interval of the quantity lies
against 0. For regret and CRPS, "below" means `dlm mixture`'s is the lower. For a quantity less a
level, "below" means under the level. The pilot column is the pilot's mean and 95 % interval.

### P1. Against the static arms

The claim's comparisons. The DLM beats the engine's static arm, its CVaR plan and PyMC-Marketing's
static model, and ties Meridian.

| Group | k | Quantity | Pilot | Scored | Prediction | Chance |
|---|---|---|---|---|---|---|
| drifting | 0 | regret, dlm mixture less stacked | -0.184 [-0.344, -0.024], n 50 | 500 | below | 0.94 |
| drifting | 0 | regret, dlm mixture less cvar 0.1 | -0.157 [-0.305, -0.010], n 50 | 500 | below | 0.92 |
| drifting | 0 | regret, dlm mixture less pymc-marketing | -0.048 [-0.128, +0.032], n 50 | 500 | below | 0.71 |
| drifting | 0 | regret, dlm mixture less meridian | +0.014 [-0.136, +0.164], n 50 | 500 | across | 0.44 |

### P2. Against PyMC-Marketing's time-varying models

The DLM loses to the shared multiplier, the setting the claim reads, and beats the per-channel one.

| Group | k | Quantity | Pilot | Scored | Prediction | Chance |
|---|---|---|---|---|---|---|
| drifting | 0 | regret, dlm mixture less pymc-marketing shared | +0.021 [-0.045, +0.086], n 50 | 500 | above | 0.50 |
| drifting | 0 | regret, dlm mixture less pymc-marketing channel | -0.038 [-0.093, +0.017], n 50 | 500 | below | 0.77 |

### P3. Against Orbit

The DLM ties Orbit.

| Group | k | Quantity | Pilot | Scored | Prediction | Chance |
|---|---|---|---|---|---|---|
| drifting | 0 | regret, dlm mixture less orbit | +0.010 [-0.193, +0.213], n 50 | 500 | across | 0.44 |

### P4. Each law, against the static arms

The DLM beats every static arm on `step` and `season` and ties them all on `static`; elsewhere the
verdicts are mixed. Most chances here are near a half: ten pilot worlds a law say little.

| Group | k | Quantity | Pilot | Scored | Prediction | Chance |
|---|---|---|---|---|---|---|
| walk | 0 | regret, dlm mixture less stacked | +0.040 [-0.354, +0.435], n 10 | 100 | across | 0.43 |
| walk | 0 | regret, dlm mixture less cvar 0.1 | +0.023 [-0.256, +0.302], n 10 | 100 | across | 0.43 |
| walk | 0 | regret, dlm mixture less pymc-marketing | -0.055 [-0.307, +0.197], n 10 | 100 | below | 0.45 |
| walk | 0 | regret, dlm mixture less meridian | +0.105 [-0.244, +0.454], n 10 | 100 | above | 0.52 |
| trend | 0 | regret, dlm mixture less stacked | +0.011 [-0.077, +0.100], n 10 | 100 | across | 0.42 |
| trend | 0 | regret, dlm mixture less cvar 0.1 | +0.081 [-0.244, +0.406], n 10 | 100 | above | 0.48 |
| trend | 0 | regret, dlm mixture less pymc-marketing | -0.013 [-0.033, +0.008], n 10 | 100 | below | 0.76 |
| trend | 0 | regret, dlm mixture less meridian | +0.293 [-0.207, +0.792], n 10 | 100 | above | 0.74 |
| step | 0 | regret, dlm mixture less stacked | -0.585 [-1.199, +0.029], n 10 | 100 | below | 0.91 |
| step | 0 | regret, dlm mixture less cvar 0.1 | -0.741 [-1.230, -0.252], n 10 | 100 | below | 0.99 |
| step | 0 | regret, dlm mixture less pymc-marketing | -0.176 [-0.532, +0.181], n 10 | 100 | below | 0.67 |
| step | 0 | regret, dlm mixture less meridian | -0.377 [-0.829, +0.074], n 10 | 100 | below | 0.87 |
| season | 0 | regret, dlm mixture less stacked | -0.298 [-0.666, +0.069], n 10 | 100 | below | 0.86 |
| season | 0 | regret, dlm mixture less cvar 0.1 | -0.156 [-0.376, +0.064], n 10 | 100 | below | 0.81 |
| season | 0 | regret, dlm mixture less pymc-marketing | -0.026 [-0.125, +0.073], n 10 | 100 | below | 0.49 |
| season | 0 | regret, dlm mixture less meridian | -0.063 [-0.256, +0.130], n 10 | 100 | below | 0.54 |
| revert | 0 | regret, dlm mixture less stacked | -0.089 [-0.315, +0.137], n 10 | 100 | below | 0.60 |
| revert | 0 | regret, dlm mixture less cvar 0.1 | +0.007 [-0.133, +0.147], n 10 | 100 | across | 0.43 |
| revert | 0 | regret, dlm mixture less pymc-marketing | +0.029 [-0.052, +0.111], n 10 | 100 | above | 0.57 |
| revert | 0 | regret, dlm mixture less meridian | +0.112 [-0.025, +0.249], n 10 | 100 | above | 0.86 |
| static | 0 | regret, dlm mixture less stacked | -0.022 [-0.265, +0.222], n 10 | 100 | across | 0.43 |
| static | 0 | regret, dlm mixture less cvar 0.1 | -0.009 [-0.167, +0.150], n 10 | 100 | across | 0.43 |
| static | 0 | regret, dlm mixture less pymc-marketing | +0.003 [-0.015, +0.021], n 10 | 100 | across | 0.41 |
| static | 0 | regret, dlm mixture less meridian | +0.014 [-0.104, +0.132], n 10 | 100 | across | 0.42 |

### P5. The no-test clause

Without a test, a plan keeps the clause on a law unless its interval less the status quo's lies
above 0. The DLM keeps it pooled and on every law but `trend`, where it breaks it.

| Group | k | Quantity | Pilot | Scored | Prediction | Chance |
|---|---|---|---|---|---|---|
| walk | 0 | regret, dlm mixture less status quo | +0.047 [-0.252, +0.346], n 10 | 100 | kept | 0.60 |
| trend | 0 | regret, dlm mixture less status quo | +0.200 [-0.228, +0.629], n 10 | 100 | broken | 0.65 |
| step | 0 | regret, dlm mixture less status quo | -0.555 [-1.032, -0.077], n 10 | 100 | kept | 0.99 |
| season | 0 | regret, dlm mixture less status quo | -0.112 [-0.348, +0.124], n 10 | 100 | kept | 0.93 |
| revert | 0 | regret, dlm mixture less status quo | +0.029 [-0.096, +0.153], n 10 | 100 | kept | 0.54 |
| static | 0 | regret, dlm mixture less status quo | -0.021 [-0.171, +0.130], n 10 | 100 | kept | 0.80 |
| drifting | 0 | regret, dlm mixture less status quo | -0.078 [-0.227, +0.070], n 50 | 500 | kept | 0.94 |

### P6. With four tests

The DLM reads no test, and the tools read theirs at their dates: at k = 4 it loses to every arm that
reads them, and still beats the status quo.

| Group | k | Quantity | Pilot | Scored | Prediction | Chance |
|---|---|---|---|---|---|---|
| drifting | 4 | regret, dlm mixture less status quo | -0.078 [-0.227, +0.070], n 50 | 500 | below | 0.66 |
| drifting | 4 | regret, dlm mixture less stacked | +0.251 [+0.057, +0.445], n 50 | 500 | above | 0.97 |
| drifting | 4 | regret, dlm mixture less cvar 0.1 | +0.130 [-0.041, +0.301], n 50 | 500 | above | 0.81 |
| drifting | 4 | regret, dlm mixture less pymc-marketing | +0.279 [+0.089, +0.470], n 50 | 500 | above | 0.98 |
| drifting | 4 | regret, dlm mixture less pymc-marketing shared | +0.277 [+0.081, +0.474], n 50 | 500 | above | 0.98 |
| drifting | 4 | regret, dlm mixture less pymc-marketing channel | +0.162 [+0.002, +0.322], n 50 | 500 | above | 0.91 |
| drifting | 4 | regret, dlm mixture less meridian | +0.166 [-0.020, +0.353], n 50 | 500 | above | 0.86 |
| drifting | 4 | regret, dlm mixture less orbit | +0.284 [+0.104, +0.464], n 50 | 500 | above | 0.99 |

### P7. PyMC-Marketing's multiplier past the history

PyMC-Marketing's time-varying models plan the quarter on their HSGP multiplier extended past the
history, where no data reach it. The arm records the process as its optimiser evaluates it
(`multiplier`, bench `1f4161c`): the lengthscale, in weeks, and each channel's mean multiplier on
the history's last week and the quarter's 13. The multiplier's mean over the history is 1 by
construction. Two quantities a world, on the drifting laws at k = 0: the lengthscale less the
quarter's 13 weeks, and the quarter's mean multiplier, over its channels and weeks, less 1.

The pilot's records carry no multiplier: its values here are the arm's at `1f4161c`, fitted again on
the pilot's 50 drifting worlds at k = 0 from their seeds (Pins). Every other item reads the pilot's
own records. The refits read a lengthscale of 1.42 weeks at the median shared and 1.51 per channel,
and above 13 weeks in 2 and 2 worlds of 50. A channel's multiplier a week into the quarter
correlates with the history's last week at 0.591 shared and 0.680 per channel on average, and at
0.013 and 0.018 by the quarter's last week. The quarter's mean multiplier is 1.033 and 1.022 at the
median, against 0.834 and 0.950 on the history's last week. Both models are predicted to forget the
history's last week within the quarter, and to plan it on a multiplier above the history's mean.

| Group | k | Quantity | Pilot | Scored | Prediction | Chance |
|---|---|---|---|---|---|---|
| drifting | 0 | lengthscale, pymc-marketing shared less 13 | -10.443 [-11.571, -9.314], n 50 | 500 | below | 1.00 |
| drifting | 0 | multiplier, pymc-marketing shared less 1 | +0.087 [+0.008, +0.166], n 50 | 500 | above | 0.93 |
| drifting | 0 | lengthscale, pymc-marketing channel less 13 | -9.870 [-11.880, -7.860], n 50 | 500 | below | 1.00 |
| drifting | 0 | multiplier, pymc-marketing channel less 1 | +0.167 [+0.062, +0.272], n 50 | 500 | above | 0.99 |

### P8. ROI coverage on the walk

Every arm's central 90 % of a channel's window ROI holds the truth less often than 90 % on `walk`,
but Meridian's, which holds it more often.

| Group | k | Quantity | Pilot | Scored | Prediction | Chance |
|---|---|---|---|---|---|---|
| walk | 0 | roi_covered, stacked less 0.9 | -0.167 [-0.386, +0.052], n 10 | 100 | below | 0.84 |
| walk | 0 | roi_covered, dlm mixture less 0.9 | -0.200 [-0.335, -0.065], n 10 | 100 | below | 0.99 |
| walk | 0 | roi_covered, pymc-marketing less 0.9 | -0.333 [-0.530, -0.137], n 10 | 100 | below | 0.99 |
| walk | 0 | roi_covered, pymc-marketing shared less 0.9 | -0.200 [-0.335, -0.065], n 10 | 100 | below | 0.99 |
| walk | 0 | roi_covered, pymc-marketing channel less 0.9 | -0.233 [-0.392, -0.074], n 10 | 100 | below | 0.98 |
| walk | 0 | roi_covered, meridian less 0.9 | +0.067 [-0.009, +0.142], n 10 | 100 | above | 0.89 |

### P9. CRPS on the walk and the step

The DLM's forecast beats Orbit's on both laws, the engine's static arm's on `step` and
PyMC-Marketing's per-channel one's on `walk`. Against PyMC-Marketing's static and shared models the
chances are near a half.

| Group | k | Quantity | Pilot | Scored | Prediction | Chance |
|---|---|---|---|---|---|---|
| walk | 0 | crps, dlm mixture less stacked | -0.003 [-0.017, +0.012], n 10 | 100 | below | 0.42 |
| walk | 0 | crps, dlm mixture less pymc-marketing | +0.004 [-0.007, +0.015], n 10 | 100 | above | 0.57 |
| walk | 0 | crps, dlm mixture less pymc-marketing shared | -0.004 [-0.016, +0.008], n 10 | 100 | below | 0.55 |
| walk | 0 | crps, dlm mixture less pymc-marketing channel | -0.021 [-0.047, +0.005], n 10 | 100 | below | 0.86 |
| walk | 0 | crps, dlm mixture less orbit | -0.054 [-0.087, -0.021], n 10 | 100 | below | 0.99 |
| step | 0 | crps, dlm mixture less stacked | -0.014 [-0.030, +0.001], n 10 | 100 | below | 0.91 |
| step | 0 | crps, dlm mixture less pymc-marketing | +0.005 [-0.011, +0.021], n 10 | 100 | above | 0.52 |
| step | 0 | crps, dlm mixture less pymc-marketing shared | -0.006 [-0.024, +0.013], n 10 | 100 | below | 0.51 |
| step | 0 | crps, dlm mixture less pymc-marketing channel | -0.008 [-0.026, +0.010], n 10 | 100 | below | 0.65 |
| step | 0 | crps, dlm mixture less orbit | -0.044 [-0.075, -0.013], n 10 | 100 | below | 0.98 |

### P10. Failures, fits past their bounds and the second reading

Few fits fail. The DLM's fits stay within their bounds, the static arm's pass them in about one
world in eight, and nearly every sampling tool's fit passes them. In the second reading those plans
stand at the status quo's regret: the DLM beats each static arm, and ties Orbit, whose variational
fit the bounds do not read.

| Group | k | Quantity | Pilot | Scored | Prediction | Chance |
|---|---|---|---|---|---|---|
| drifting | 0 | past, stacked | 7 of 50 | 500 | 36-120 of 500 | 0.90 |
| drifting | 0 | past, dlm mixture | 0 of 50 | 500 | 0-19 of 500 | 0.95 |
| drifting | 0 | past, pymc-marketing | 49 of 50 | 500 | 461-499 of 500 | 0.93 |
| drifting | 0 | past, pymc-marketing shared | 50 of 50 | 500 | 481-500 of 500 | 0.95 |
| drifting | 0 | past, pymc-marketing channel | 50 of 50 | 500 | 481-500 of 500 | 0.95 |
| drifting | 0 | past, meridian | 49 of 50 | 500 | 461-499 of 500 | 0.93 |
| drifting | 0 | failure, stacked | 0 of 50 | 500 | 0-19 of 500 | 0.95 |
| drifting | 0 | failure, dlm mixture | 0 of 50 | 500 | 0-19 of 500 | 0.95 |
| drifting | 0 | failure, pymc-marketing | 0 of 50 | 500 | 0-19 of 500 | 0.95 |
| drifting | 0 | failure, pymc-marketing shared | 0 of 50 | 500 | 0-19 of 500 | 0.95 |
| drifting | 0 | failure, pymc-marketing channel | 0 of 50 | 500 | 0-19 of 500 | 0.95 |
| drifting | 0 | failure, meridian | 0 of 50 | 500 | 0-19 of 500 | 0.95 |
| drifting | 0 | failure, orbit | 0 of 50 | 500 | 0-19 of 500 | 0.95 |
| drifting | 0 | regret, dlm mixture less stacked, second reading | -0.213 [-0.380, -0.047], n 50 | 500 | below | 0.97 |
| drifting | 0 | regret, dlm mixture less cvar 0.1, second reading | -0.159 [-0.309, -0.009], n 50 | 500 | below | 0.92 |
| drifting | 0 | regret, dlm mixture less pymc-marketing, second reading | -0.078 [-0.227, +0.070], n 50 | 500 | below | 0.66 |
| drifting | 0 | regret, dlm mixture less meridian, second reading | -0.079 [-0.228, +0.069], n 50 | 500 | below | 0.67 |
| drifting | 0 | regret, dlm mixture less pymc-marketing shared, second reading | -0.078 [-0.227, +0.070], n 50 | 500 | below | 0.66 |
| drifting | 0 | regret, dlm mixture less pymc-marketing channel, second reading | -0.078 [-0.227, +0.070], n 50 | 500 | below | 0.66 |
| drifting | 0 | regret, dlm mixture less orbit, second reading | +0.010 [-0.193, +0.213], n 50 | 500 | across | 0.44 |

### P11. Meridian and the box

Meridian's optimiser plans outside the box in nearly every world-rung, and the plan is moved into
it.

| Group | k | Quantity | Pilot | Scored | Prediction | Chance |
|---|---|---|---|---|---|---|
| every law | 0 | moved, meridian | 60 of 60 | 600 | 580-600 of 600 | 0.96 |
| every law | 4 | moved, meridian | 60 of 60 | 600 | 580-600 of 600 | 0.96 |

### P12. Claimed less realised gain

Every arm but Meridian claims more gain than it realises. The DLM's 90 % interval of its gain holds
the realised gain in 43–89 of 100 worlds on `walk` and on `trend`, under its level.

| Group | k | Quantity | Pilot | Scored | Prediction | Chance |
|---|---|---|---|---|---|---|
| walk | 0 | uplift, dlm mixture less 0 | +2.236 [+0.131, +4.341], n 10 | 100 | above | 0.94 |
| trend | 0 | uplift, dlm mixture less 0 | +2.709 [+1.395, +4.022], n 10 | 100 | above | 1.00 |
| step | 0 | uplift, dlm mixture less 0 | +3.425 [+1.459, +5.391], n 10 | 100 | above | 0.99 |
| season | 0 | uplift, dlm mixture less 0 | +1.983 [+0.311, +3.655], n 10 | 100 | above | 0.96 |
| revert | 0 | uplift, dlm mixture less 0 | +1.886 [+0.744, +3.029], n 10 | 100 | above | 0.99 |
| static | 0 | uplift, dlm mixture less 0 | +1.835 [+1.116, +2.555], n 10 | 100 | above | 1.00 |
| walk | 0 | covered, dlm mixture | 7 of 10 | 100 | 43-89 of 100 | 0.90 |
| trend | 0 | covered, dlm mixture | 7 of 10 | 100 | 43-89 of 100 | 0.90 |
| step | 0 | covered, dlm mixture | 8 of 10 | 100 | 54-95 of 100 | 0.91 |
| season | 0 | covered, dlm mixture | 10 of 10 | 100 | 82-100 of 100 | 0.96 |
| revert | 0 | covered, dlm mixture | 9 of 10 | 100 | 66-99 of 100 | 0.93 |
| static | 0 | covered, dlm mixture | 9 of 10 | 100 | 66-99 of 100 | 0.93 |
| drifting | 0 | uplift, stacked less 0 | +1.781 [+1.419, +2.144], n 50 | 500 | above | 1.00 |
| drifting | 0 | uplift, cvar 0.1 less 0 | +1.056 [+0.671, +1.442], n 50 | 500 | above | 1.00 |
| drifting | 0 | uplift, dlm plug-in less 0 | +2.418 [+1.752, +3.085], n 50 | 500 | above | 1.00 |
| drifting | 0 | uplift, dlm cvar 0.1 less 0 | +0.800 [+0.381, +1.218], n 50 | 500 | above | 1.00 |
| drifting | 0 | uplift, pymc-marketing less 0 | +1.892 [+1.483, +2.302], n 50 | 500 | above | 1.00 |
| drifting | 0 | uplift, pymc-marketing shared less 0 | +1.586 [+1.166, +2.005], n 50 | 500 | above | 1.00 |
| drifting | 0 | uplift, pymc-marketing channel less 0 | +1.538 [+1.149, +1.926], n 50 | 500 | above | 1.00 |
| drifting | 0 | uplift, meridian less 0 | -0.006 [-0.067, +0.054], n 50 | 500 | across | 0.44 |
| drifting | 0 | uplift, orbit less 0 | +2.609 [+2.096, +3.122], n 50 | 500 | above | 1.00 |

## True by construction

- The walk law is a DLM's own state law: a random walk in the log of each channel's effect. The
  other laws, and every law's ranges, are the bench's own.
- Beneath the drift each world is a Track M v2 drawn world, whose media are PyMC-Marketing's
  default class, `tanh` with a normalised kernel, and one of the engine's.
- The best plan knows the law and the state at the history's end.
- Meridian counts nothing after the quarter, and Robyn plans one steady week.
- The DLM reads no lift test, so its plan at k = 4 is its plan at k = 0 (P6).

## Compute

From the pilot's records, measured under the pilot's load, so a planning figure and not a timing:

- CPU seconds a world-rung: the static arm's fit and plans 260; PyMC-Marketing 117 static, 298
  shared and 603 per channel; Meridian 286; Orbit 81. A world, both rungs: the DLM's one fit and
  three plans 777.
- At 600 worlds and two rungs: about 680 core-hours, 460 of them the tools'. Robyn's 50 worlds add
  95–195.
- Peak memory a process: the DLM 4.0 GiB at the median and 5.4 at most; the static arm 0.9; Meridian
  more than 5 GiB at k = 0, so it runs under an 8 GiB cap.
- PyMC-Marketing's arms run with pytensor's compile cache off, so a fit compiles afresh. One
  pilot world's per-channel fit took about 975 CPU seconds with the cache off and 687 with a warm
  one, under load; the pilot's figures above ran with the cache. Budget about 40 % more CPU for each
  of the three arms' fits, most of it compiling: about 815 core-hours in all.
