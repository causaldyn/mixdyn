# 0097. The noise mixture is the margin regression's default

Status: proposed, 2026-11-21; moves the default of `noise` in
`mixdyn.models.margin_mmm.fit_margins` and `fit_curved_margins` from None to `NoiseMixture()`
(`mixdyn.models.margin_mmm.MIXTURE`), and supersedes ADR 0091's decision "Defaults are unchanged";
`noise=None` fits as before, to the bit, and no arm's plan changes; bounds the mixture's memory,
to the same bits, and raises `MemoryError` where the memory left cannot take a fit (Consequences).

## Context

ADR 0087 recorded a defect of the margin regression without a noise mixture: the ridge on the
trend, which carries the sales' level, lets a channel's column absorb part of the level. ADR 0091
removed it under the mixture, a flat prior on the nuisance columns among its four parts, and kept
the default for the readers frozen on the regression without one. So a caller who names no `noise`
reads the defective regression. The README's example names the mixture.

**The symptom.** The README's history: 104 weeks of three channels' i.i.d. uniform weekly spend,
each through a geometric adstock of retention 0.5 over 8 weeks, margins of 1.5, 3.0 and 0.5, white
noise of deviation 25 and a level of 5000, 200 noise deviations. The default read margins of 4.48,
8.30 and 0.20, deviations 1.01, 2.37 and 0.86, at mean retentions of 0.90, 0.90 and 0.20 and an
AR(1) coefficient of 0.53. The mixture read 1.44, 2.19 and 0.21, deviations 0.22, 0.49 and 0.61.

**The cause is the prior on the level, not the error model.** Without a mixture each coefficient's
prior is centred at 0, of precision `PRIOR` times its column's whitened sum of squares, the trend's
B-spline basis included. The basis sums to 1, so the trend carries the level, and the prior pulls
the level towards 0. At one retention combination and one AR(1) coefficient, sales `y + c` give
the coefficients of `y` plus `c (e - (X' S^-1 X + Lambda)^-1 Lambda e)`, `e` the trend's
coefficients of the constant. The channels move by the second term, which is 0 where
`Lambda e = 0`, under a flat prior on the trend, and not otherwise. A channel's column, its mean
the spend's, takes up part of the level, a smoother column more: the evidence moves to the high
retentions. The level left over is slow, and the AR(1) coefficient reads it as persistence. Four
reads show it.

- *The level alone moves the default's margins.* The same history at a level of 0 reads 1.41, 2.20
  and 0.16 at an AR(1) coefficient of 0.00; at 1000, 1.64, 3.43 and 1.26; at 10 000, 5.36, 9.86
  and 0.05 at 0.65. The mixture reads 1.44, 2.19 and 0.21 at every level, its margins at 0 and at
  5000 within 1.4e-11 of each other and their covariance within 2.4e-10.
- *At the truth's retentions the ridge alone is off.* At retentions of 0.5 and an AR(1) coefficient
  of 0, the posterior means are 2.40, 4.58 and 2.17 under the ridge, 1.50, 2.45 and 0.15 under a
  flat prior on the nuisance columns, and 1.59, 2.58 and 0.16 by least squares. Maxima, in exact
  rational arithmetic on the same design, gives the first two within 1.2e-12 and 2.4e-11 of
  numpy's. It proves the identity above on a design of two basis columns that sum to 1 and one
  channel: the shift is linear in `c`, and 0 with the basis columns' prior precisions at 0.
- *The pull's size.* The prior's term at the level alone, `b' Lambda b` at `b = c e`, over the
  noise's residual sum of squares, is `PRIOR (c / sigma)^2` times the weeks' mean of
  `sum_j B_j(t)^2`, 0.46 on this trend: 0.73 at a level of 1000, 40 noise deviations, and 18 at
  5000. The bench's worlds, whose baseline deviates by 0.064 of the level (ADR 0091), sit near
  0.1.
- *Over 20 seeds of the README's history*, at four levels and under white errors and an AR(1) of
  0.6 of the same innovations, the median margin read over the truth's and the share of the seeds
  within two deviations of the truth, a channel each:

| errors | level | default: median over the truth | default: within 2 sd | mixture: median over the truth | mixture: within 2 sd |
|---|---|---|---|---|---|
| white | 0 | 0.92, 0.95, 0.85 | 0.85, 0.95, 1.00 | 0.98, 1.02, 1.22 | 0.95, 0.95, 1.00 |
| white | 1000 | 1.30, 1.28, 4.66 | 0.85, 0.95, 0.85 | the same | the same |
| white | 5000 | 3.06, 2.88, 5.49 | 0.15, 0.40, 0.95 | the same | the same |
| white | 20 000 | 5.47, 4.29, 5.51 | 0.40, 0.85, 1.00 | the same | the same |
| AR(1) 0.6 | 0 | 0.96, 0.95, 1.16 | 0.95, 0.80, 1.00 | 0.98, 0.96, 1.50 | 0.90, 0.90, 1.00 |
| AR(1) 0.6 | 1000 | 1.31, 1.22, 3.76 | 0.85, 0.85, 0.90 | the same | the same |
| AR(1) 0.6 | 5000 | 2.75, 2.21, 5.29 | 0.25, 0.55, 1.00 | the same | the same |
| AR(1) 0.6 | 20 000 | 4.94, 4.00, 8.44 | 0.50, 0.90, 1.00 | the same | the same |

  The third channel's margin, 0.5, is about one deviation, so its ratio is noisy under both.

**A fix on the default path is not open.** The fix is a flat prior on the nuisance columns. With
the AR(1) kept at the residual's moment it is ADR 0091's first alternative, which that ADR's
freezing rule did not admit: it lost on family 12, 0.1562 against 0.1409 a euro. And the default
path is the one the frozen readers read: the `margin` arm's `margin`, `margin guarded`, `margin
ranged` and `margin curved`, and the tested rung's `anchored`, `anchored curved` and `anchored
held` with the composites `gs`, `gs curved` and `gs held`, the measured claim's arm at the tested
rung. Any change to it moves them all. The mixture is measured: `margin held` (ADR 0092), the
claim's arm with no lift test, reads under it.

## Decision

- **`noise` defaults to `MIXTURE`**, `NoiseMixture()`, in `fit_margins` and `fit_curved_margins`.
  `noise=None` fits the regression without a mixture, to the bit.
- **The readers frozen on the regression without a mixture name it.** `scorecard.arms.margin`'s
  fit and `scorecard.arms.anchored`'s call `fit_margins(..., noise=None)`. `margin curved` passed
  None before, and `margin curved noise` and `margin held` pass `NOISE_MIXTURE`. `project` keeps
  no `noise`: it is the regression without a mixture at one combination, as `anchored curved` reads
  it, and its docstring says so.
- **CHOICE: both functions.** The module states `fit_curved_margins` as `fit_margins`' regression
  with each column saturated. Defaults that differ would read the line and the curve off two
  regressions.
- **CHOICE: a module constant in the signatures**, `MIXTURE`: one frozen object every default fit
  shares, and no call in a default argument (ruff's B008).
- **The module's docstring states the defect** of the regression without a mixture, with the
  README's history and the identity above, beside the readers that keep it.

## Consequences

- *Every fit that names no `noise` moves.* The module paths are not public names
  (`docs/api/index.md`), so a minor release may move them; the changelog states it under
  *Changed*.
- *What did not move.* Every test that read the regression without a mixture names `noise=None`,
  its expectations as they were: the research code's margins on six of M2's worlds, the fit
  assembled as it was before the mixture to the bit, and every other test of that regression in
  `tests/test_margin_mmm.py`. With the arms calling the default instead, 22 of the arms'
  research-parity tests fail, 6 of `margin` and 16 of `anchored` and `anchored curved`
  (`tests/scorecard/`, which run where the bench is importable): those fixtures guard the names.
  So every arm, `margin held` and `gs held` among them, plans as it was measured, to the bit.
- *A caller frozen on the regression without a mixture that names no `noise` moves with this
  change.* One written on another branch before it names `noise=None` when it is merged; where it
  is an arm with research-parity fixtures, they fail until it does.
- *Histories the default now refuses.* Under the mixture, nuisance columns the weeks read do not
  tell apart raise `DegenerateFitError` where the ridge fitted them (ADR 0091): a price that never
  moves, a price that is a straight line in the week, a promotion flag that is its promoted weeks'
  dummies' sum. Such a history is read without that control, or with `noise=None`.
- *Memory.* The mixture fits every error model at every combination: `124 5^C` fits for `C`
  channels under the default grids. Stacked whole, their covariances alone took `124 5^C C^2`
  doubles, and the fit held three to five copies of them: one fit of 104 weeks peaked over the
  interpreter at 1.9 GiB at six channels, 2.9 GiB for the curved fit, and passed 8 GiB at seven,
  where the process was stopped. So each fit now keeps its log evidence, weight, error model,
  margins and the channels its prior sets, `40 + 17 C` bytes; the regressions are solved a block
  of combinations at a time; and the fits' second moments, `8 C^2` bytes each, are held only where
  they take at most 1 GiB, six channels or fewer under the default grids, and the memory left takes
  them. Otherwise every fit is read again for their sum, a block at a time, each block's sum
  carried into the next as the held sum carries a fit's. numpy's sum over the fits adds them one
  after another, so the mixture is the same to the bit either way: held and read again, at two to
  six channels, straight and curved, every field of the fit is the stacked fit's, and a test reads
  the fits both ways on every run. The same fit now peaks at 0.8 GiB at six channels and 1.3 GiB
  at seven, under a cap of 4 GiB, at the cost of the regressions read twice. Eight channels take
  8.6 GiB by the estimate below, the fits' rows 8 GiB of it, and under the cap the fit raises; the
  memory falls with the error models' count, or with `noise=None`, where the regression without a
  mixture took 1.2 GiB at eight.
- *The memory guard.* Before the fit takes the memory it estimates it; where the estimate is more
  than the memory left, the system's available memory or what a cgroup v2 limit on the process
  leaves, it raises `MemoryError` naming the channels, the fits, the memory and the remedies, never
  leaving the process to the out-of-memory killer. The estimate bounds the arrays numpy traces at
  0.4 to 0.7 of it from two to six channels. Outside Linux the memory left is not known and the fit
  is not guarded. No time was measured here; ADR 0091 put the curved fit at 0.30 to 0.39 CPU-s a
  world under the mixture and 0.22 to 0.24 without, at three channels.
- *The mixture's limit stays* (ADR 0091): the error model's posterior means describe the mixture
  planned on, not the noise alone. On the README's history, whose noise is white, the posterior
  mean of `rho` is 0.49; over the 20 seeds above it is 0.54 under white errors and 0.67 under an
  AR(1) of 0.6.
- *The tests* (`tests/test_margin_mmm.py`). By default both functions fit under `NoiseMixture()`,
  every field to the bit; on the README's history the default's margins at a level of 0 and of
  5000 agree within 1e-9 and lie within two deviations of the truth, and without a mixture the
  level moves the first two margins by more than two deviations, past 2.5 times the truth. On the
  previous default both tests fail: the first margin reads 4.476 by default where the mixture reads
  1.441, and 1.405 at a level of 0.

## Alternatives considered

- **A flat nuisance prior on the default path**: ADR 0091's first alternative, which its freezing
  rule did not admit, and a change to every frozen reader's regression.
- **The default's scope in the docstring and the README alone**: the default would keep reading the
  level into the margins on any history whose level is large against its noise, as a sales
  history's usually is, and a caller avoids it only by reading the docstring.
- **The sales centred before the regression**: the margins shift-invariant, the ridge still on the
  trend's variation and on the channels' means; another regression no run measured, which moves
  the frozen readers as the first does.
- **The mixture's moments accumulated in one pass**, the weights rescaled to the largest log
  evidence seen as each block arrives: no regression read twice, but another order of the
  arithmetic of `margin held`'s regression, whose plans would move in their last bits.
- **A guard alone**, the fits stacked whole: it would refuse seven channels on a machine of 8 GiB
  free, and a mixture of seven channels fits in 1.3 GiB.
