# 0086. The margin arm: linear margins read by regression, shrunk, and planned at a corner

Status: proposed, 2026-11-17; adds `mixdyn.models.margin_mmm`, `mixdyn.decisions.margin_plan` and
the scorecard's `margin` arm, its variants `margin`, `margin guarded` and `margin ranged`; changes
no default and no other arm's plan.

## Context

M2's no-test plans lost to Meridian (ADR 0085). A screen of k = 0 plan rules on M2's development
set, M2's scored worlds, found a linear rule that beats the engine's k = 0 candidate,
`dlm cvar 0.1`: a regression reads each channel's margin, the KPI a unit of weekly spend returns;
the margins are shrunk towards their common level by empirical Bayes; and the plan is the one a
straight-line response makes of them, a corner of the box. The rule was found with an
implementation written apart from this package, in numpy and scipy.

A rule's score is the largest mean paired difference of regret a euro, its plan's less a rival's,
over families 0, 2, 3, 5 and 7 and the rivals status quo, equal split, Meridian, PyMC-Marketing and
Orbit: below 0, the rule beats every rival on every one of those families. Measured on M2's
development set at k = 0:

| score | fit split | check split |
|---|---|---|
| the margin rule | -0.0120 | +0.0128 (family 2, Meridian) |
| `dlm cvar 0.1` | +0.0287 | +0.0399 (family 2, Meridian) |

The rule was chosen on the fit split, post hoc, among about 40 plan rules and 30 estimator
configurations, and then read once on the check split, frozen, on its 822 worlds of families 0, 2,
3, 5 and 7. On the check split its mean regret less Meridian's was -0.049, +0.013, +0.010, -0.038
and +0.008 on families 0, 2, 3, 5 and 7, the 95 % intervals of families 2, 3 and 7 including 0;
less the status quo's, -0.133, -0.032, -0.020, -0.072 and -0.019. On the check split's worlds of
family 4, whose curves are S-shaped, its mean regret was 0.340, 0.18 below Meridian's; on family
14's, every one of its 94 fits raised, and the read played the status quo.

Why a straight line can plan these worlds, as far as measured: the truth's curvature at the status
quo is 0.02-0.18, so a plan's expected worth is nearly linear in the margins across the box, and
the best plan for a linear worth is the corner on the posterior mean. Through the corner, the
worlds' true margins leave 0.002-0.007 of regret a euro. A regression's margins are Gaussian in
linear space, and shrinking them there pulls a noisy channel's estimate in more than a precise
one's, which removes the winner's curse at the corner's argmax.

The research implementation had two defects:

- **Long kernels.** It dropped the first `L - 1` weeks of the history as the adstock's burn-in.
  Family 14's exports carry a kernel of `L = 156` weeks, the whole history: one week was left, the
  AR(1) coefficient's sums over consecutive weeks were empty, 0/0, and every fit raised.
- **Degenerate fits.** Nothing stopped a fit with no residual degree of freedom or an AR(1)
  coefficient that is not a number; it raised from inside the linear algebra, on an array of NaN.

## Decision

- **`mixdyn.models.margin_mmm.fit_margins(sales, spend, promotion, price, kernel)`** reads the
  margins. The design: each channel's geometric adstock of the export's kernel length, its weights
  `r^l` normalised to sum to 1, at a retention `r` on {0.1, 0.3, 0.5, 0.7, 0.9}; a cubic B-spline
  trend with interior knots every 26 weeks, which sums to 1 and is the intercept; 3 yearly
  harmonics; the flag; the price; and a dummy for each promoted phase, a week of the year
  (`t mod 52`) whose mean price over the years is below the price's median by more than 0.15. A
  history without the flag (`promotion` None) leaves its column out, and one without the price its
  column and the promoted phases' dummies, which are read off it; the rest of the design is the
  same, and with both the design is the research implementation's, column for column. Every
  coefficient has a normal prior about 0 whose precision, `Lambda`'s diagonal, is `1e-3` times its
  column's sum of squares in the design it is fitted on, a column of 0 counting 1. The errors are
  AR(1): `phi` is the lag-1 coefficient of the residuals of the same retentions' fit with
  independent errors, clipped to [0, 0.95], and the AR(1) covariance `S`'s Cholesky factor whitens
  the design `X`. Each of the `5^C` combinations of retentions is weighted by its closed-form
  evidence with the variance at its plug-in, `-1/2 log|S| + 1/2 log|Lambda| - log|R| - n/2 log RSS`,
  `R` the Cholesky factor of `X'S^-1 X + Lambda`; the margins are the mixture's mean and covariance
  of the channels' coefficients, each combination's covariance `RSS / (n - p)` times
  `(X'S^-1 X + Lambda)^-1`. Each channel's `prior_share` is the evidence weight of the
  combinations in which the prior's precision on its coefficient, `Lambda_cc`, exceeds the data's,
  the Schur complement `1 / [(X'S^-1 X)^-1]_cc`, read as the whitened column's residual sum of
  squares on every other whitened column; it moves no margin.
- **`mixdyn.decisions.margin_plan.shrink_margins(mean, cov)`** shrinks them: `b ~ N(beta, V)`,
  `beta = mu 1 + delta`, `delta ~ N(0, tau^2 I)`, `mu` flat; `tau` is `tau_rel` times the positive
  margins' mean, `tau_rel` on 60 points even in its log over [0.01, 3], weighted by the REML
  likelihood, a half-normal prior of scale 0.5 and the log grid's Jacobian. The posterior mean is
  mixed over `tau`, and so is its covariance, which the research implementation did not compute:
  each `tau`'s, `(P / tau^2 + V^-1)^-1` with `P` the centring matrix, and the spread of the means
  about the mixture's (checked in Maxima: at one `tau` it is the law of total variance over `mu`).
  No positive margin: nothing is shrunk.
- **`mixdyn.decisions.margin_plan.plan_corner(margins, lower, upper, budget)`** plans: every
  channel at its floor, the rest of the budget to the highest margin up to its ceiling, then the
  next. Equal margins fill in the channels' order.
- **The scorecard's `margin` arm** (`scorecard.arms.margin.Margin`) runs the three on an export;
  its variant `margin` is the rule as it was found. It reads no lift test (`tests = False`): the
  harness fits it once a world, and each rung's plan is k = 0's, which the record says. It reads
  the controls `promotion` and `price` where the export holds them, and its settings name those
  read. It plans the market (`cells = False`), and refuses a geo export and its market as a fit's
  failure, which the bench scores as the status quo: the rule reads one national history, and M3
  plans the geo families with its geo candidates; the record's error says so. Where no margin is
  positive, the plan is the status quo. The record claims the plan's gain over the status quo on
  the shrunk means, `planned m'(p - s)` over the quarter and the kernel's tail, exact for
  straight-line channels whose kernel sums to 1, with the normal interval of the shrunk
  posterior's covariance at 90 %. Its flags: each channel's margin and its standard deviation
  before and after the variant's shrinkage, its posterior mean retention, its prior share and
  whether the variant held it, the posterior mean of `tau_rel`, `phi`'s mean over the
  combinations, the promoted phases read, the most likely combination's evidence weight, the rows
  read and the burn-in before them. Its settings name every constant above.
- **The guard, the variant `margin guarded`.** A channel whose spend never moves has a column the
  trend's basis already spans, and only the prior splits the level between them: on family 19
  `margin` read the flat channel's margin as the largest and filled it (Consequences). The guarded
  variant holds a channel whose margin the data do not identify at its status quo, its floor and
  its ceiling in the corner both its status quo, and leaves its margin out of the shrinkage, which
  pulls the other channels' towards their own common level; the corner spends the rest of the
  budget on them. Not identified: the combinations in which the prior's precision on its
  coefficient exceeds the data's hold more than half of the evidence weight (`prior_share` above
  0.5), collinearity with every other column counted. With no channel held, its plan is
  `margin`'s to the bit; with every channel held, or no positive margin among the others, the
  status quo. Its settings say why each channel it held was held. CHOICE: of the two definitions
  measured, this one, which the rule declared before the measurement kept if both passed.
- **The observed range, the variant `margin ranged`.** The regression reads a channel's margin off
  the weekly spend its history logged, and inside that range alone do the data say anything of
  it: a plan past the history's least or most extrapolates a straight line no week supports. On
  family 6 every plan of `margin` put a channel below its history's least (Consequences).
  `mixdyn.decisions.margin_plan.observed_range(spend, lower, upper, budget)` cuts each channel's
  box to the range of weekly spend over the export's whole history, the floor the larger of the
  box's and the history's least, the ceiling the smaller of the box's and the history's most; the
  variant is `margin` with the corner in that box, and its settings name the planning range. A
  status quo that is a mean of the history's weeks, the last year's in every family but the
  twins', lies inside each channel's range and its box, so the floors sum below its weekly total
  and the ceilings above it. The twins set theirs apart, (100, 50) a week; on their 200 pilot
  worlds the floors summed at least 82.8 a week below that budget and the ceilings at least 55.8
  above it. An export where they do not raises `ObservedRangeError`, a `ValueError` whose message
  gives the cut box: the harness records the variant's plan as failed, the other variants
  standing, and the bench scores it as the status quo. Read by a rule declared before any of its
  plans was computed, it is acceptable as `margin`'s replacement at k = 0 (Consequences). The
  variant is kept either way; which variant M3 scores is not this record's decision.
- **CHOICE: the weekly budget is the status quo's weekly total**, as the rule was found with. The
  bench's budget is the quarter's weeks times it in every family; an export whose budget is
  another is refused.
- **CHOICE: the record holds no response.** The bench reads a record's response for its
  return-on-investment scores alone, and maps a geometric kernel through a `tanh` or a Hill
  (`scorecard.arms.mapped`); a straight line is neither. The plan's regret and its claim are scored
  without one, and the record says why under `n/a`.
- **The long kernel.** CHOICE: each channel's spend before the history is imputed at its mean over
  the history's first 13 weeks, a steady state at its opening level, and the burn-in is
  `min(L - 1, 13)` weeks. From week `L` on, the adstock is the history's own to the bit, whatever
  the imputation, so for every kernel of 14 weeks or less the fit is the research
  implementation's.
- **The degenerate fit.** A design with no residual degree of freedom, a residual whose lagged sum
  of squares is 0, a whitened residual sum of squares that is not positive, or a mixture that is
  not finite raises `mixdyn.models.margin_mmm.DegenerateFitError`, a `ValueError` that names the
  case. The harness records it as the fit's failure; no NaN reaches a plan.
- **Defaults are unchanged.** No other arm plans, scores or claims otherwise.

## Consequences

- *Parity.* On six of M2's worlds at k = 0 in the development set's fit split, one each of
  families 0, 2, 3, 4, 5 and 7, the margins, their covariance, the posterior mean retentions, the
  most likely combination and its weight, `phi`, the shrunk means and the plan are the research
  implementation's: to 1e-9 relative, the plan exactly (`tests/fixtures/margin_worlds.json`, its
  expected values written by that implementation). On the machine that wrote the fixture they
  agreed to the bit. A family 14 world, on which that implementation raised, fits after 13 weeks of
  burn-in on 143 rows.
- *The screen* (M2's development set, fit split, k = 0, the archived exports, no rival refitted;
  the question and the rule written before it ran). Run through the scorecard's arm interface and
  each plan scored by the bench's scorer, on 40 worlds, 8 of each of families 0, 2, 3, 5 and 7
  drawn by a seed, the arm's regret was the research implementation's on every world, within
  4.7e-16 a euro and equal on 33; no fit failed, and the scorer moved no plan. On 10 worlds of
  family 14 drawn the same way, exploratory, every fit ran, after 13 weeks of burn-in on 143 rows.
  Mean regret a euro on those 10 worlds:

  | the margin arm | `stacked` | Meridian | the status quo |
  |---|---|---|---|
  | 0.085 | 0.442 | 0.390 | 1.002 |

  The other three are M2's recorded plans. The arm's regret was 0 on 5 of the 10 worlds, and it
  beat Meridian's on 7. Ten worlds read for a screen are no claim.
- *M2's development set, check split, family 14* (k = 0, the arm declared before the split was
  read, read once): on its 94 worlds no fit failed, and the mean regret was 0.3165 a euro. The
  arm's mean regret less each rival's, with the 95 % interval where one was computed:

  | rival | mean difference |
  |---|---|
  | the status quo | -0.404 [-0.594, -0.236] |
  | Meridian | -0.080 [-0.199, +0.046] |
  | PyMC-Marketing | -0.025 [-0.134, +0.081] |
  | `stacked` | -0.072 [-0.218, +0.066] |
  | Orbit | -0.283 |
  | `dlm cvar 0.1` | -0.191 |
  | Robyn, on its 20 worlds | -0.190 |
- *The other non-geo families* (M2's development set, fit split, k = 0, the archived exports, no
  rival refitted; the question and the rule written before it ran). M3's no-test clause asks, on
  every family its claim covers, that the claim's arm less the status quo have a paired interval
  not above 0. A family is safe here where the arm's regret less the status quo's has a 95 %
  interval, 2000 resamples of the worlds within each environment, whose lower end is at most 0.
  Beside it, the rival the arm lost to most, by the mean difference, among the status quo, equal
  split, Meridian, PyMC-Marketing, Orbit, `stacked` and `dlm cvar 0.1`, to be read against a
  margin of 0.05. No fit failed. Families 12 and 13 have no worlds in M2's development set and
  were not read.

  | family | worlds | mean regret | less the status quo | the rival lost to most | safe |
  |---|---|---|---|---|---|
  | 1, drift | 78 | 0.308 | -0.093 [-0.184, -0.005] | Meridian, -0.034 | yes |
  | 4, S-curves | 120 | 0.257 | -0.388 [-0.459, -0.321] | `stacked`, +0.076 [+0.025, +0.129] | yes |
  | 6, outside the range | 120 | 0.390 | +0.119 [+0.047, +0.189] | equal split, +0.145 | no |
  | 6, cliff | 40 | 0.471 | +0.209 [+0.087, +0.344] | Meridian, +0.219 | no |
  | 6, saturation | 40 | 0.291 | -0.013 [-0.122, +0.092] | equal split, +0.032 | yes |
  | 6, both | 40 | 0.407 | +0.161 [+0.033, +0.287] | equal split, +0.194 | no |
  | 14, Robyn's | 146 | 0.264 | -0.308 [-0.424, -0.194] | Meridian, -0.126 | yes |
  | 15, siMMMulator | 53 | 0.057 | -0.057 [-0.079, -0.033] | `stacked`, -0.015 | yes |

  Family 15's exports hold no control; its row is read with the controls read where held, and
  before that every one of its fits was refused. On family 6 every plan put a channel below the
  least weekly spend its history showed, the corner's floor, and the loss to the status quo came
  in the environments whose curves fall off a cliff below the history's range, not in the one
  whose curves saturate early above it. `margin` does not guard against it, and on family 6 it
  fails the clause; `margin ranged` keeps every plan inside the history's range, and passes it
  (*The observed range*).
- *The guard* (M2's development set, fit split, k = 0, all 1073 worlds of families 0, 2, 3, 5 and
  7 from the archived exports, and family 19's 300 pilot worlds, 100 an environment; the two
  definitions and the rule written before any guarded plan was computed). Definition 1, prior
  dominance, is the variant's. Definition 2 held a channel whose weekly spend's coefficient of
  variation over the history was below `sqrt(1e-3 / (1 - 1e-3)) = 0.0316`, where the prior's
  precision over the data's, `1e-3 (1 + 1/CV^2)` with the intercept alone in the design, crosses
  1. A definition passed if its plans were `margin`'s on at least 95 % of the 1073 worlds, its
  mean regret on each of the five families within 0.002 of `margin`'s, and it never planned family
  19's flat channel above its status quo.

  | plan | equal to `margin`'s | mean regret moved | family 19, flat channel filled | family 19, mean regret |
  |---|---|---|---|---|
  | `margin` | | | 285 of 300 | 0.239 |
  | prior dominance | 1071 of 1073 | -0.0006 on family 7, 0 elsewhere | 0 | 0.173 |
  | the spend's CV | 1073 of 1073 | 0 | 0 | 0.173 |

  Both passed, and the declared choice keeps prior dominance. Each held family 19's flat channel
  on all 300 worlds and no other channel: the flat channel's prior share was 1 on every one, a
  moving channel's 0. Prior dominance also held one channel on 2 of family 7's 52-week worlds,
  whose short histories leave its column close to the other columns: on one the regret rose from
  0 to 0.025, on the other it fell from 0.184 to 0.058. On the 1073 worlds `margin`'s scored
  regret was the research implementation's within 9.5e-16 a euro, and equal on 859.

  Read after the choice, and so no part of it, on the screen's worlds of families 1, 4, 6, 14
  and 15: prior dominance held no channel on families 1, 4 and 6; on family 14 it held one on 7
  of 146 worlds, and the mean regret rose by 0.0007; on family 15, whose spends move least, their
  coefficients of variation 0.10 and up, and whose kernel is the whole history, it held one on 39
  of 53, and the mean regret rose by 0.043, from 0.057 to 0.100, the regret less the status quo's
  moving to -0.015 [-0.032, +0.003]. The spend's CV would have held no channel on any of those
  worlds. On family 15, `margin` is the better plan of the two.
- *The observed range* (M2's development set, fit split, k = 0, all 1073 worlds of families 0, 2,
  3, 5 and 7 and the screen's 517 of families 1, 4, 6, 14 and 15 from the archived exports; the
  rule written before any ranged plan was computed). `margin ranged` was acceptable as `margin`'s
  replacement at k = 0 if (1) its score was at most `margin`'s on the same worlds plus 0.005;
  (2) on family 6, all of it and each environment alone, its regret less the status quo's had a
  95 % interval whose lower end was at most 0; and (3) families 1, 4, 14 and 15 stayed safe by the
  screen's clause. All three held. (1) Its score was -0.0126, `margin`'s -0.0120, both on family 3
  against Meridian. (2) Less the status quo, family 6 read +0.014 [-0.041, +0.069]: cliff +0.066
  [-0.029, +0.172], saturation -0.061 [-0.148, +0.023], both +0.035 [-0.064, +0.134]. (3) Families
  1, 4, 14 and 15 read -0.096 [-0.182, -0.013], -0.353 [-0.420, -0.289], -0.300 [-0.412, -0.190]
  and -0.037 [-0.049, -0.023]. No plan failed, and the harness read every ranged plan inside the
  history's range.

  | family | worlds | mean regret, `margin` | mean regret, `margin ranged` | plans moved | channels at the history's least / most |
  |---|---|---|---|---|---|
  | 0 | 266 | 0.1315 | 0.1370 | 199 | 199 / 0 |
  | 1 | 78 | 0.3083 | 0.3055 | 43 | 43 / 0 |
  | 2 | 234 | 0.1561 | 0.1567 | 51 | 51 / 0 |
  | 3 | 213 | 0.1632 | 0.1626 | 65 | 65 / 0 |
  | 4 | 120 | 0.2573 | 0.2923 | 79 | 79 / 0 |
  | 5 | 186 | 0.1493 | 0.1476 | 65 | 65 / 0 |
  | 6 | 120 | 0.3897 | 0.2844 | 120 | 176 / 0 |
  | 7 | 174 | 0.1592 | 0.1578 | 107 | 106 / 1 |
  | 14 | 146 | 0.2638 | 0.2723 | 70 | 70 / 0 |
  | 15 | 53 | 0.0569 | 0.0775 | 53 | 68 / 38 |

  A plan moved where it moved by more than 1e-9 of the weekly budget, which it did on exactly the
  worlds where a channel sat at a bound of its history inside the box; on 221 more it moved by
  at most 1e-12 of it, rounding. The bound that binds is the history's least, the corner's floor;
  the most binds on 38 channels of family 15, whose spends move least, and on one of family 7.
  The range gains 0.105 a euro on family 6 and costs 0.035 on family 4, whose curves are S-shaped,
  0.021 on family 15, 0.008 on family 14 and 0.006 on family 0. On family 6 its mean regret less
  Meridian's, the equal split's and `dlm cvar 0.1`'s was +0.036, +0.040 and +0.001 over the
  family, +0.076, +0.067 and +0.063 on cliff, -0.028, -0.016 and -0.081 on saturation, and
  +0.061, +0.069 and +0.020 on both, every interval including 0; over the family, less
  PyMC-Marketing's and Orbit's, -0.19 and -0.31, neither interval including 0. On cliff and both
  its mean regret still exceeded the status quo's. Read without a rule on family 19's 300
  pilot worlds, it held the flat channel at the one level its history logged on all of them, above
  its status quo on 121 by at most 1.4e-15 of it, the status quo's mean of equal weeks rounding;
  its mean regret was 0.174, `margin guarded`'s 0.173 and `margin`'s 0.239. On the 1373 worlds the
  guard's measurement shares with this one, the plans of `margin` and `margin guarded` were the
  same to the bit.
- *The claim.* On the 40 worlds the claimed gain exceeded the realised one by 0.23 a euro on
  average, and its 90 % interval held the realised gain on all 40; on family 14's 10 it fell short
  of it by 0.57 and held it on 7.
- *The cost.* No sampler: a fit is `5^C` generalised least-squares fits, each with its evidence.
  On a laptop loaded by other work, so to be read as an order of magnitude, a fit took 0.10-0.14
  CPU-s on 156 weeks (families 0, 2, 3, 4, 5 and 14) and 0.04-0.07 on family 7's shorter
  histories; a world's whole process under the harness, its imports included, 3.7-4.2 CPU-s, and
  its scope's memory peaked at 168 MiB. The prior shares add `C` least-squares solves to each
  combination: over the guard's 1373 fits, under the same load, a fit took a median of 0.16 CPU-s
  on 156 weeks and 0.09 on family 7's histories.
- *The prior's pull.* The prior pulls each column's contribution towards 0 by `1e-3` of its squared
  norm, so the more a world's sales and spend sit above their noise, the more it moves the margins:
  on a world made by the model, its sales 200 noise deviations above 0, it moved them by up to 4.8
  of their standard deviations, and at a prior of 1e-9 the world's retentions and margins were
  found (`tests/test_margin_mmm.py`). On the fixture's seven worlds the plans at 1e-9 were the plans
  at 1e-3, though the margins moved, family 3's from (2.71, 4.69, 2.16) to (14.05, -16.19, 1.57).
  The rule is ported as it was measured, prior included; another prior is another rule.
- *What it cannot read.* The geo families (8, 9, 10) are refused. Family 15's and family 17's
  exports hold no control and are read without one; family 18's third control, `searches`, is not
  read. A channel whose spend never moves (family 19) has a column the trend already holds, which
  only the prior separates: on the first pilot world of each of family 19's environments the flat
  channel's margin was the largest of the three (8.9, 14.5 and 28.3), and so was its shrunk
  margin, and the corner filled it first. `margin` keeps it; `margin guarded` holds the channel at
  its status quo.
- *The tests* (`tests/test_margin_mmm.py`, `tests/test_margin_plan.py`,
  `tests/scorecard/test_margin_arm.py`): the parity above; a kernel as long as the history, made
  and family 14's; the burn-in and the imputation; the adstock against the history extended at the
  imputed level; the promoted phases; a history without a control fitting without its columns, and
  at a negligible prior finding a world's margins; the prior share of a spend that never moves, 1,
  and of moving ones, 0, and the Schur complement against the cross-products' inverse; the typed
  error on a history too short for the design and on sales the design fits exactly, and the input
  refused; the shrinkage against the law of total variance at one `tau`; the corner in the box and
  on the budget, filling in order and breaking ties by the channels' order; the arm's record, its
  claim against the scorecard's reading of straight-line channels, an export without a control and
  family 17's fitting in every variant, the geo export and its market refused, its other refusals
  and a run end to end beside the built-in arms; and the guarded variant planning as `margin` with
  no channel held, holding family 19's flat channel, holding a channel at its status quo while the
  others are shrunk and cornered alone, and playing the status quo with every channel held; the
  observed range cutting each box to the spend its history logged, raising its typed error where
  the cut box holds no plan on the budget, and refusing input it does not read with an untyped
  one; and the ranged variant planning the corner in the cut box, inside the logged range, where
  `margin` plans the research plan, and failing alone on an export whose cut box holds no plan.

## Alternatives considered

- **Shrinking in log space with the delta method's variances.** The shrinkage factor there,
  `tau^2 b^2 / (tau^2 b^2 + V)`, rises with the estimate, so the largest and noisiest margin is
  shrunk least. Measured on the fit split: +0.069, against the rule's -0.012.
- **Anchoring at the status quo**, by fractional Kelly steps, z-gates or a CVaR's anchor. In these
  worlds the status quo's regret is 0.20-0.34 a euro and its order of the channels is arbitrary;
  every anchor lost on the fit split.
- **For the long kernel:** refusing a kernel the history cannot burn in, so family 14 always plays
  the status quo; cutting the kernel at 14 weeks, a model other than the export's kernel; or no
  spend before the history, which the worlds do not have: at retention 0.9 the weeks before the
  history carry 23 % of the kernel's weight in the first row read after 13 weeks.
- **A straight-line response in the record**, which the bench's mapping would need a new curve to
  read; neither the regret nor the claim reads it.
- **Holding a channel by its spend's coefficient of variation**, below 0.0316, where the prior's
  precision crosses the data's with the intercept alone counted. It passed the same rule, held
  the same channels on family 19 and none on the 1073 worlds, and costs no solve; but it sees no
  collinearity with the trend's shape, the season, the controls or the other channels, and the
  rule declared before the measurement kept prior dominance where both passed. Read after the
  choice, it would have held nothing on family 15, where prior dominance cost 0.043 a euro; a
  switch to it is a new rule, to be declared and read on worlds the choice did not see.
