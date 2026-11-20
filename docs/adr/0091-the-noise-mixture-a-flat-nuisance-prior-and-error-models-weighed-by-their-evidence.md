# 0091. The noise mixture: a flat nuisance prior and error models weighed by their evidence

Status: proposed, 2026-11-17; adds `mixdyn.models.margin_mmm.NoiseMixture`, the keyword `noise` of
`fit_margins` and `fit_curved_margins`, `MarginFit.share` and `MarginFit.promoted`, and the `margin`
arm's variant `margin curved noise`; changes no default and no other variant's plan.

## Context

`margin curved` (ADR 0087) reads each channel's margin off a regression whose errors are an AR(1)
at the residual's lag-1 moment, beside a trend with knots every 26 weeks, under a ridge prior on
every column. ADR 0087 recorded two defects of that regression and left them: the ridge on the
trend, which carries the sales' level, lets a channel's column absorb part of the level; and a
promoted phase inside the burn-in leaves a column of zeros that the ridge's floor keeps.

A study written apart from this package, in numpy and scipy over this package's adstock, design
columns and shrinkage, looked at where the regression's error comes from before any plan's regret
was read. On the bench's worlds the baseline's stochastic part has a deviation of 0.064 of the
level and autocorrelations of 0.80, 0.69, 0.52 and 0.31 at lags 1, 2, 4 and 8, and the worlds hold
no white observation noise; the 26-week trend with an AR(1) reads a coefficient of about 0.43 off
it. The promoted weeks' innovations have about 3 times the other weeks' variance. With the level
in the ridge, family 12's raw margins had a median of 1.67, 3.31 and 2.57 times the truth's a
channel, against 1.02, 0.98 and 0.68 on the media alone; a flat prior on the nuisance columns
removes that.

The study then declared a freezing rule, ran it once over 29 variants of the regression on the
tuning worlds (M2's fit split at k = 0 and 70 + 70 worlds of families 12 and 13 outside the
pilot's), and read the one variant it froze once, at k = 0, on M2's check split (1365 worlds) and
on the pilot's 30 + 30 worlds of families 12 and 13. The planner was `margin curved`'s
throughout, unchanged. Its declared decision held. The frozen variant beside `margin curved`,
mean regret a euro:

| read | the frozen variant | `margin curved` |
|---|---|---|
| check split, the score of ADR 0087 | -0.0311 (family 7, Meridian) | -0.0155 (family 2, Meridian) |
| the pilot, family 12, the worst mean less a rival's | +0.0180 (Orbit) | +0.0433 (Orbit) |
| the pilot, family 13, the worst mean less a rival's | -0.0255 (the equal split) | +0.0048 (the equal split) |
| the pilot, family 12, mean regret | 0.1978 | 0.2231 |
| the pilot, family 13, mean regret | 0.1150 | 0.1453 |

On the check split, less `margin curved`, it regretted 0.174 a euro less on family 14
[-0.240, -0.111] and 0.030 less on family 2 [-0.046, -0.013]; no family's interval lay above 0,
the largest mean cost +0.008 on families 0 and 4. The no-test clause held on families 1, 4, 6, 14
and 15. Its gain on family 12 is not established: against Orbit on the pilot's 30 worlds the
interval is [-0.118, +0.189], and against `margin curved` on the same worlds the mean is -0.025
[-0.071, +0.019]. Family 4's S-curves stay 0.18 behind PyMC-Marketing's and Orbit's plans, as
under `margin curved`. The M3 selection's predictive was 1.000, 1.000, 1.000, 1.000 and 0.999 on
families 0, 2, 3, 5 and 7, 0.387 on family 12 and 0.912 on family 13, against `margin curved`'s
0.286 and 0.674 on the last two. This is the third frozen k = 0 rule to read these worlds, so the
predictives are optimistic, as ADR 0087 says of its own. Of 16 other ways from the regression's
coefficients to the planned margins, two passed the rule's guards, with a third of family 12's
gain: the shrinkage stays `shrink_margins`.

## Decision

- **`mixdyn.models.margin_mmm.NoiseMixture`** is one settings object for the regression under a
  mixture of error models: `knot_every` (52), the AR(1) coefficients `persistence` (0.3, 0.6, 0.8,
  0.9, 0.95, 0.98), the AR(1)'s shares of the errors' variance `share` (0.5, 0.7, 0.85, 0.95, 1),
  the promoted weeks' variance ratios `promoted` (1, 2, 4, 8), and `white`, white errors once
  more: 31 pairs times 4 ratios, 124 error models. `fit_margins(..., noise=None)` and
  `fit_curved_margins(..., noise=None)` take one; without one they fit as before, to the bit.
- **Under a mixture the regression differs in four parts**, its mean the same.
  (1) The trend's interior knots lie every `knot_every` weeks.
  (2) The nuisance columns, the trend, the season, the controls and the promoted weeks' dummies,
  carry a flat prior, and a nuisance column that is 0 on every week read is dropped. Each
  channel's coefficient keeps a Gaussian prior, its precision `PRIOR` times the whitened sum of
  squares of the channel's straight adstock column times `(kappa / (1 + kappa))^4`, the squared
  slope of its curve at a steady status quo: every curve puts the line's prior on the margin there.
  (3) The errors are a stationary AR(1) of coefficient `rho` plus white noise, the AR(1)'s share
  of their variance `omega`; the weeks are whitened by the Cholesky factor of that covariance, so
  each whitened week reads the weeks before it alone.
  (4) The errors' variance on a promoted phase's weeks is `h` times the other weeks'.
- **Every error model and every retention combination is weighted by its evidence** under a
  uniform prior over them, the nuisance coefficients and the noise's scale integrated out:
  `-1/2 log|S| + 1/2 sum_c log Lambda_c - 1/2 log|X' S^-1 X + Lambda| - (n - q)/2 log RSS`, `q` the
  nuisance columns kept. It is the same on any common scale of `S`
  (`validation/margin_noise.mac`), so the error models are compared as their covariances are
  written. The margins' posterior is the mixture's mean and covariance over all of them.
- **`MarginFit` reports the error model's posterior means**: `phi` that of `rho`, and two new
  fields, `share` that of `omega` and `promoted` that of `h`; without a mixture both are 1. `top`
  is the retention combination of the most weight over the error models, and a channel's
  `prior_share` weighs the models in which the prior's precision exceeds what the data add to it,
  `1 / [(X' S^-1 X + Lambda)^-1]_cc - Lambda_c`.
- **A channel the nuisance columns span has a margin of exactly 0.** Where a channel's columns
  lie in the nuisance columns' span at every retention, as a spend that never moves lies in the
  trend's, the flat prior lets the nuisance coefficients take the channel's whole column: the data
  say nothing of its coefficient, whose posterior is its prior, mean 0, and it moves no other
  channel's. Floating point computes a residue in place of that 0, about 1e-12 of the other
  margins and of either sign. So the fit reads the design first. A channel whose columns'
  relative distance from the span, the largest over the retentions, is at most `SPANNED`, 1e-8,
  has a coefficient of exactly 0 in every model, and `MarginFit.spanned` names it; without a
  mixture no channel is named, the ridge on the nuisance columns setting such a margin. The
  variant holds a spanned channel at its status quo, its box a point, and its settings say why.
- **CHOICE: one settings object, not a keyword a part.** The four parts were frozen together and
  the evidence's degrees of freedom, `n - q`, hold only under the flat prior with the zero columns
  dropped; no caller is given a way to take one part without the others. The grids are the
  object's fields because the evidence is defined for any of them, and a test fits one error model
  at a time through them.
- **The variant `margin curved noise`** (`scorecard.arms.margin`) is `margin curved` with its
  regression under `NoiseMixture()`: the shrinkage, the curves, the box and the water-filling are
  `margin curved`'s, to the line of code. Its flags add `ar share` and `promoted variance`, and
  `phi` is the posterior mean of `rho`; its settings name the trend's knots, the nuisance prior and
  the grids. The name adds one word to the variant it changes, as `margin guarded`, `margin ranged`
  and `margin curved` add one to `margin`; it has no comma, since variants' names are joined by
  commas where they are listed.
- **`margin curved` stays**, and so does the regression every other reader runs: the tested rung's
  `anchored curved` and `gs curved` (ADR 0089) were frozen on the regression without a mixture and
  read it as one moment; the mixture was not read through them.
- **Defaults are unchanged.** `fit_margins`' and `fit_curved_margins`' arithmetic without a
  mixture is the code it was; its mixture's weights moved into a private `_weigh` that both paths
  share, the operations and their order kept.

## Consequences

- *Parity.* The variant's records were made by the arm's harness from the archived exports and
  scored by the bench's `score_world` against each world's archived export, the frozen rule's plan
  of the same export beside them in the same call. They give the frozen rule's regret on every
  world the rule was read on, to at most 2.5e-13 a euro: 1.5e-13 on the 1730 tuning worlds,
  2.5e-13 on M2's 1365 check-split worlds and 0 on the pilot's 60. The two plans differ by at most
  4.3e-11 of a weekly spend near 100. The rule's regret as run here is the one its study recorded,
  to the bit, on all 3155 worlds. No record failed and the scorer moved no plan.
- *What did not move.* On the same 3155 worlds `margin curved`'s plan is the one its own parity
  stored before this change, to the bit, and `margin ranged`'s regret on the 1425 worlds with a
  stored one is that one. `anchored curved` and `gs curved`, read again on the pilot's 60 worlds
  and on every 10th check-split world at the tested rung (104), gave every regret, route and plan
  of the read before this change. On the margin and the tested-rung fixtures' 15 worlds and on 8
  synthetic histories, every field of `fit_margins`, `fit_curved_margins` and `project` without a
  mixture equals the previous code's to the bit.
- *The departures from the code the rule was frozen in, none of which moved a regret.* (1) That
  code dropped an error model whose residual sum of squares was not positive from the mixture;
  here, as in `fit_margins`, it raises `DegenerateFitError` and fails the variant. No world read
  had one. (2) Nuisance columns that the weeks read do not tell apart, as a price that never moves
  beside the trend, leave the flat prior improper. That code met them in a Cholesky factorisation,
  which fails or returns a factor of rounding as the last pivot's sign falls; here the nuisance
  columns' rank is read first and the fit raises `DegenerateFitError`. No world read had such
  columns. (3) It asserted 13 planned weeks; here they are the export's, as in `margin curved`.
  (4) It fitted the line beside the curve; here only the curve is fitted, as in `margin curved`,
  and the line's column gives the prior's precision.
- *A spanned channel: what the code the rule was frozen in left to rounding.* The shrinkage's
  scale is the mean of the positive margins, so the residue's sign decided whether a 0 entered
  it. It was measured on 60 worlds of a family with a channel whose spend never moves (family
  19, 20 worlds of each of its three environments), none of them a world the rule was read on.
  The residue was positive on 25. The plan with its sign set positive and the plan with it set
  negative differ on 25, by up to 7.3 % of the weekly budget. Moving the flat spend by 1e-9 of
  itself along a pattern, and against it, gave the variant before this hold one sign each way
  and plans as far apart; with the margin read as 0 the two plans agree within 2.8e-10 of the
  weekly budget on all 60.
  Two readings of the 0 were measured. In the first the channel stays in the shrinkage with a
  margin of 0, which is not positive, and its box is a point. In the second the other channels
  are shrunk without it. They differ on 15 of the 60 worlds, by up to 1.1 % of the weekly
  budget. The first was taken: it is the frozen rule in exact arithmetic, and the second is
  another rule, which nothing read. The variant's plan is then that code's with the residue's
  sign set negative and its adstock held inside the spend's range (ADR 0087), on 60 of 60 worlds
  within 1.4e-14 of the weekly budget. Against that code as it ran here it differs on 8 of the
  60: on 7 by the residue's sign, and on 1 by that code's fallback of the whole box on a last bit
  of the flat channel's adstock.
  The criterion is the design's distance from the span and not the channel's prior share. The
  share is 1 on those flat channels and 0.53 at most on every other channel read, but it
  depends on the prior's scale, which follows the error model (the limit below); the distance
  reads the columns alone, and a distance of 0 is exactly the condition under which the margin
  is 0. On the flat channels it is 2.9e-16 at most; on every other channel of the 3215 worlds
  read it is 0.080 at least. So on the 3155 worlds the rule was read on no channel is spanned,
  and the variant's plan is the one before the hold, to the bit.
- *A limit of the rule as frozen: the weights over the error models compare evidence only where
  the channels' prior is nearly flat against the data.* A channel's prior precision is `PRIOR`
  times `(kappa / (1 + kappa))^4` times the whitened sum of squares of its straight column,
  uncentred. So the prior's term in the penalised residual sum of squares, `b' Lambda b`, is about
  that share of the whitened energy of the channel's whole contribution, its mean level included.
  A persistent error model shrinks a level's whitened norm, by `(1 - rho) / (1 + rho)` under an
  AR(1) alone, and so widens the prior. Where a channel's contribution is large against the noise,
  the term is not small against the residual sum of squares, and the evidence prefers the error
  model that widens the prior, whatever the noise is. Two regimes were measured.
  On a synthetic history of white errors with margins of 2.0, 0.5 and 1.2 and sales near 1000, the
  largest contribution is about 40 noise deviations and the prior's term about 1.4 times the
  residual sum of squares under white errors: the posterior mean of `rho` is 0.98. With `PRIOR`
  at 1e-9 it is 0.08, and 0.94 on the same history with an AR(1) of 0.9
  (`tests/test_margin_mmm.py`).
  On the bench's worlds the prior is nearly flat. On 6 fit-split worlds an environment, no regret
  read, the prior's term was 0.002 to 0.008 of the penalised residual sum of squares under the
  error model of the most weight on families 0 to 6 and 14, 0.03 on family 7's 52-week histories
  and 0.026 on family 15. The posterior mean of `rho` at `PRIOR` against the one at 1e-9 moved by
  at most 0.05 on families 0 to 6 and 14, their means 0.71 to 0.86 both ways; by up to 0.39 on
  one 78-week world of family 7, the family's means 0.55 against 0.45; and on family 15, whose
  errors are nearly white, from 0.08 to 0.27. On the seven fixture worlds the posterior means are
  0.51 to 0.91 for `rho`, 0.85 to 0.93 for `omega` and 3.2 to 7.9 for `h`.
  It is not fixed here: the frozen rule is what was read on the check split. A prior scale that
  does not follow the error model, one set by the columns after the nuisance columns are
  partialled out for example, is work after M3. Until then the flags `phi`, `ar share` and
  `promoted variance` describe the mixture that was planned on, not the noise alone.
- *The cost.* The laptop was loaded by other work, so these numbers are an order of magnitude.
  The variant's plan, its regression under the 124 error models and its water-filling, took 0.30
  to 0.39 CPU-s a world at the median of the three reads and 1.29 at most over the 3155 worlds;
  `margin curved`'s took 0.22 to 0.24 and 0.52. Its record, the arm's shared linear fit counted,
  took 0.46 to 0.59 and 1.74. Each error model is whitened once, and every retention combination
  is read off one table of the whitened columns' cross-products.
- *What it cannot read.* As `margin curved`; and an export whose nuisance columns the weeks read
  do not tell apart fails this variant alone, where the ridge of `margin curved` fits it. A
  price that never moves, a price that is a straight line in the week and a promotion flag that
  is its promoted weeks' dummies' sum are such exports, and each is tested. A flat prior depends
  on the nuisance columns' span alone, so reducing the columns to a basis of their span would
  plan such a history; that is left for later.
- *The two defects ADR 0087 recorded* are absent under a mixture: the level carries no prior, and
  a column of zeros is dropped and not counted in the residual degrees of freedom. Without a
  mixture both stand, as the readers frozen on them need.
- *The tests* (`tests/test_margin_mmm.py`, `tests/scorecard/test_margin_arm.py`). On the seven
  fixture worlds the mixture's margins, their covariance, the error model's posterior means, the
  shrunk margins and the plan are those of the code the rule was frozen in
  (`tests/fixtures/margin_noise_worlds.json`). Each error model's fit is the generalised least
  squares under its covariance inverted whole, and at an AR(1) alone it is Prais and Winsten's
  regression under a flat nuisance prior, the evidence equal across the two scales. With every week
  promoted the variance ratio scales the whole covariance and moves no fit's evidence. A zero
  column is dropped and the fit is that of the history without it, to the bit. A channel whose
  spend never moves is named spanned and has a margin of exactly 0, of its prior's variance; it is
  spanned up to a move of 1e-9 of itself and not at 1e-6; and the variant's plan on a flat channel
  is the same whichever sign rounding would leave its margin. At a negligible
  prior the mixture tells white errors, an AR(1) and wide promoted weeks apart. Without a mixture
  the fit is assembled from each combination's regression in the arithmetic it had, to the bit.
  Each of these tests fails on two mutants of what it names. `validation/margin_noise.mac` proves
  the covariance of an AR(1) beside white noise, its determinant and its factor, the evidence's
  scale invariance and the weeks' own scales, under Maxima 5.46 and 5.50.

## Alternatives considered

- **A flat nuisance prior alone**, the AR(1) kept at the residual's moment: on the tuning worlds
  it gained on families 0 and 14 and lost on family 12 (0.1562 against 0.1409 a euro), and the
  freezing rule did not admit it.
- **The AR(1) plus white noise without the variance ratio**: the rule did not admit it (family 12
  at 0.1378 on its 70 tuning worlds, against the frozen 0.1232). **With the ratio's grid ending at
  4** the rule admitted it and froze the wider grid (family 12 at 0.1250).
- **The coefficient held at 0.6 or below**: the best mean gain over M2's families and no cost on
  family 0, but its gain on family 12 changed sign between the two halves of the tuning worlds,
  which the rule refuses.
- **Another shrinkage of the mixture's margins**: another half-normal scale, a fixed `tau`, a
  log-normal prior, or none. Two of 16 passed the rule's guards, with a third of family 12's gain.
- **Replacing `margin curved`'s regression.** The tested rung's readers were frozen on it, and the
  mixture was not read through them: a new variant leaves every frozen reading as it is.
