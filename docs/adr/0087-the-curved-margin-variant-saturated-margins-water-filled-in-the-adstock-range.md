# 0087. The curved margin variant: saturated margins, water-filled inside the adstock range

Status: proposed, 2026-11-17; adds `mixdyn.models.margin_mmm.fit_curved_margins`,
`mixdyn.decisions.margin_plan.water_fill` and `mixdyn.decisions.margin_plan.adstock_range`, and the
`margin` arm's variant `margin curved`; changes no default and no other variant's plan.

## Context

`margin ranged` (ADR 0086) plans the corner that a straight line makes of each channel's margin.
Its weak cells were the concave worlds. On the M3 pilot's 30 + 30 worlds of families 12
(Meridian's generator) and 13 (PyMC-Marketing's) at k = 0, its worst mean regret less a rival's was
+0.1694 a euro (family 12, Orbit) and +0.0881 (family 13, the equal split). On M2's check split at
k = 0 it lost to Meridian on family 6, whose curves change outside the history's range, by +0.0538
[+0.003, +0.103].

Two readings, written apart from this package in numpy and scipy, fixed the design before any
plan's regret was read:

- **Where the regret comes from.** The worlds were 70 + 70 of families 12 and 13 outside the
  pilot's and 60 of family 0. There the corner on each channel's true margin at the status quo
  regretted 0.011, 0.038 and 0.032 a euro on families 0, 12 and 13, against `margin ranged`'s
  0.140, 0.186 and 0.145. The true curves planned on `margin ranged`'s margins regretted 0.138,
  0.152 and 0.131. So the margins' ranking carries most of the regret, not the corner's shape. A
  straight line fitted to a concave curve reads the curve's mean slope over the history's adstock.
  That mean slope exceeds the slope at the status quo the more the adstock spreads, so a bursty
  channel is over-rated. A curved regression reads the slope at the status quo.
- **What one history identifies.** The half-saturation was left free and weighed by the evidence,
  under a flat prior on the nuisance columns. At the bench's noise the posterior elasticity at the
  status quo, at a half-saturation common to the channels, had medians 0.70, 0.54 and 0.64 on
  families 0, 12 and 13; the truth's medians are 0.965, 0.44 and 0.92. With a half-saturation for
  each channel, its correlation with the truth within a family ran from -0.15 to +0.31. A rule
  cannot read each world's curvature off its history: the curvature it plans on is its prior's. So
  the half-saturation is fixed.

The screen read 38 variants of the margin rule, each reducing to `margin ranged` at an infinite
half-saturation. Its worlds were M2's fit split at k = 0, 1590 worlds of families 0, 2, 3, 5 and 7
and of the no-test clause's families 1, 4, 6, 14 and 15, and the 70 + 70 worlds above. The variants:
the curved margins water-filled along their curves at a common relative half-saturation `kappa` of
1, 2, 4, 8, 16 or 32; the linear margins on the curves alone; the curved margins at the corner
alone; and `kappa` weighed by the flat-nuisance evidence. Each ran in `margin ranged`'s box and in
that box cut further to each channel's adstock range.

A rule declared before any of their regrets was read froze one variant. It had to meet four bounds:
(i) a fit-split score at most `margin ranged`'s + 0.005; (ii) on each clause family, the lower end
of its difference to the status quo at most 0; (iii) a mean regret below `margin ranged`'s by at
least 0.03 on each of families 12 and 13 (D12, D13); (iv) family 6 less Meridian at most +0.005.
Among the variants that met all four, the rule took the largest min(D12, D13), and within 0.005 of
it the lowest family-6 cell. Ten variants met all four. The best min(D12, D13) was +0.0428; seven
variants lay within 0.005 of it; of those, the curved margins at `kappa` 16 in the adstock-cut box
read family 6 lowest, and the rule froze them:

| fit split and the 70 + 70 worlds, k = 0 | score (worst cell) | D12 [95 %] | D13 [95 %] | family 6 less Meridian |
|---|---|---|---|---|
| `margin ranged` | -0.0126 (family 3, Meridian) | 0 | 0 | +0.0363 |
| curved, `kappa` 16, adstock-cut box (frozen) | -0.0259 (family 7, Meridian) | +0.0455 [+0.004, +0.089] | +0.0415 [+0.013, +0.072] | -0.0576 |
| curved, `kappa` 16, `margin ranged`'s box | -0.0256 | +0.0367 | +0.0398 | -0.0202 |
| curved, `kappa` 8, adstock-cut box | -0.0262 | +0.0428 | +0.0480 | -0.0519 |
| curved, `kappa` 4, adstock-cut box | -0.0253 | +0.0426 | +0.0441 | -0.0422 |
| `kappa` by the evidence, adstock-cut box | -0.0210 | +0.0345 | +0.0286 | -0.0401 |
| linear margins on the `kappa` 16 curves, adstock-cut box | -0.0197 | +0.0450 | +0.0186 | -0.0386 |
| curved `kappa` 16 margins at the corner, adstock-cut box | -0.0242 | +0.0297 | +0.0244 | -0.0673 |

Then the frozen rule was read once, at k = 0, on M2's check split (1365 worlds) and on the pilot's
30 + 30 worlds of families 12 and 13. Its declared decision held. On the check split, each
family's mean regret less Meridian's:

| check split, k = 0 | worlds | the frozen rule | `margin ranged` |
|---|---|---|---|
| family 0 | 174 | -0.0557 | -0.0428 |
| family 2 | 177 | -0.0155 | +0.0122 |
| family 3 | 168 | -0.0292 | +0.0090 |
| family 5 | 156 | -0.0534 | -0.0394 |
| family 7 | 147 | -0.0253 | +0.0044 |
| family 6, outside the range | 120 | -0.0235 | +0.0538 |

The score, the largest mean difference to the status quo, the equal split, Meridian,
PyMC-Marketing or Orbit over families 0, 2, 3, 5 and 7, was -0.0155 (family 2, Meridian), against
`margin ranged`'s +0.0122 and the declared bound +0.0172. The no-test clause held: less the status
quo, the lower ends on families 1, 4, 6, 14 and 15 were -0.236, -0.214, -0.098, -0.408 and -0.035.

| the pilot, k = 0 | the frozen rule's worst mean less a rival | `margin ranged`'s | declared bound |
|---|---|---|---|
| family 12 (30 worlds) | +0.0433 (Orbit) | +0.1694 (Orbit) | +0.139 |
| family 13 (30 worlds) | +0.0048 (the equal split) | +0.0881 (the equal split) | +0.058 |

The curve costs where the truth is S-shaped or the kernel outruns the history. On the check split
the rule regretted 0.115 a euro more than `margin ranged` on family 4, whose curves are S-shaped
Hill, Weibull and logistic curves, and 0.17 more than PyMC-Marketing's and Orbit's plans there. It
regretted 0.114 more on family 14, Robyn's worlds, whose kernel spans the history, and 0.024 more
on family 15. The M3 selection's predictive (`scorecard.development.predictive`), read on the
check split and the pilot's worlds, was 1.000, 0.982, 1.000, 1.000 and 0.998 on families 0, 2, 3,
5 and 7, 0.286 on family 12 and 0.674 on family 13. Several frozen rules have read the check split
and the pilot's worlds, the later ones written after the earlier reads. So these predictives are
optimistic, and M3's fresh worlds are the test.

## Decision

- **`mixdyn.models.margin_mmm.fit_curved_margins(sales, spend, promotion, price, kernel,
  status_quo, planned, half_saturation=16)`** is `fit_margins`' regression with each channel's
  adstock column saturated, `g(a) = a / (1 + a / K_c)`, `K_c = 16 s_c` (`HALF_SATURATION`). At a
  steady status quo its elasticity is 16/17 whatever the channel's level; an infinite
  half-saturation is the line. Every retention combination is fitted and weighed by its evidence
  as `fit_margins` fits and weighs it, under the same ridge prior. Each combination's margin at the
  status quo is `beta_c G'_cr(s_c)`, with `G_cr(p) = 1/n sum_h g(carry_crh + p reach_rh)` over the
  `n` planned weeks and the kernel's tail after them. `carry` is the history carried into the
  quarter through the combination's kernel, the spend before the history imputed as `fit_margins`
  imputes it. The margins' posterior is the mixture's mean and covariance (`CurvedFit.margins`, a
  `MarginFit` with its diagnostics). Each channel's curve for planning, `CurvedFit.curve`, is the
  evidence-weighted mixture over its retentions of `G_cr / G'_cr(s_c)`, of slope 1 at the status
  quo.
- **`mixdyn.decisions.margin_plan.water_fill(margins, slope, shaped, lower, upper, budget)`**
  plans the most gain `sum_c m_c H_c(p_c)` in the box at the budget. At a price `lam`, each channel
  of positive margin spends where `m_c H'_c(p_c) = lam`, held to its box, solved by 56 bisections.
  A channel whose curve is a line, or whose margin is not positive, sits at its ceiling above the
  price and at its floor below, as the corner plans it. The price is bisected 68 times in its
  logarithm, over 80 below twice the highest margin, until the plan spends at most the budget.
  What is left goes to a line at the price, then to the channels in the order of their margins.
  Where the positive channels at their ceilings leave budget over, the rest goes by the corner;
  where no curve is curved, the plan is the corner's.
- **`mixdyn.decisions.margin_plan.adstock_range(spend, adstock, lower, upper, budget)`** cuts
  `observed_range`'s box further to each channel's range of its adstock over the history, the
  spend before it imputed. Where that box holds no plan on the budget, the observed range stands.
- **The variant `margin curved`** (`scorecard.arms.margin`) runs the curved regression on the
  export and shrinks its margins at the status quo by `shrink_margins`. It water-fills them along
  each channel's curve inside `adstock_range`'s box, each channel's adstock read at its retention
  of the most evidence weight. Where no margin is positive, it plays the status quo. Its flags are
  the curved regression's, as the margin arm flags its own, with each channel's half-saturation,
  the retention its range reads and the range. Its settings name the planner, the half-saturation
  and the planning range.
- **CHOICE: the curved regression runs in the variant's plan**, not in the arm's fit. The arm's
  variants share one fit, and its cost is charged to each of them. In the fit, the curved
  regression would double the recorded cost of `margin`, `margin guarded` and `margin ranged`, and
  a curved fit that the data leave degenerate would fail them too. In the plan, its cost and its
  failure are `margin curved`'s alone, and the record's cost still counts the shared linear fit.
- **CHOICE: the claim is read along the curves**, `planned sum_c m_c (H_c(p_c) - H_c(s_c))`: the
  plan's gain over the quarter and the kernel's tail on the shrunk margins, with the normal
  interval of their covariance at 90 %. On lines it is `margin ranged`'s claim. The rule as found
  claimed nothing. The record holds no response, as the margin arm's holds none: a channel's curve
  is a mixture over retentions, and a record's response maps one geometric kernel through a `tanh`
  or a Hill.
- **CHOICE: the planned weeks are the export's** (`planned`), 13 in every family the rule was found
  on; the rule as found took the constant.
- **`fit_margins` is unchanged to the bit.** Its loop over the combinations became the private
  `_combination` and `_mixture`, which the curved regression shares. On the margin and the
  tested-rung fixtures' 15 worlds and on 6 synthetic histories, every field of its fit equals the
  previous code's to the bit. `plan_corner`'s checks moved into a private `_box` that `water_fill`
  shares, their order and messages kept.
- **Defaults are unchanged.** No other arm or variant plans, scores or claims otherwise.

## Consequences

- *Parity.* The variant's records were made by the arm's harness from the archived exports and
  scored by the bench's `score_world` against each world's archived export. They give the frozen
  rule's regret on every world the rule was read on, to at most 1.01e-11 a euro: 8.9e-12 on the
  screen's 1730 worlds (M2's fit split and the 70 + 70 worlds), 1.01e-11 on M2's 1365 check-split
  worlds, and 3.8e-12 on the pilot's 60. No record failed and the scorer moved no plan.
- *The departures, none of which moved a regret past those differences.* (1) The rule as found
  fitted the line beside the curve and planned on the curve's fit alone; its code read the line off
  the end of its grid of half-saturations, so the grid had to end there. Here only the curve is
  fitted. (2) It ran each combination's regression on the moment matrices of the design; here the
  combinations share `fit_margins`' dense whitening, the same arithmetic to rounding. (3) It
  dropped a degenerate combination from the mixture; here, as in `fit_margins`, a degenerate
  combination raises `DegenerateFitError` and fails the variant. No world read had one. (4) The
  adstock is held in its channel's range of spend before its range is read
  (`adstock_range`). An adstock is a weighted mean of the spends, so the hold changes it by
  rounding alone, but on a channel whose spend never moves the rule as found is decided by that
  rounding: the computed adstock lands a last bit beside the constant, the cut box is then empty,
  and every channel falls back to the observed range. Which retention's adstock is read there is
  itself a tie of the evidence broken in the 12th digit. On 20 worlds of a family with such a
  channel (family 19, `flat-tv`) the rule as found fell back on 2 by that bit; with its adstock
  held the same way it is the variant's on all 20, to 7.5e-13 a euro, and the mean regret moves
  from 0.1351 to 0.1353. No world the rule was read on has such a channel: on those 3155 worlds
  the hold changed no plan by a bit.
- *The box.* On the screen's worlds the planning box was the observed range itself on 72 of
  family 2's 234 worlds, 2 of family 3's 213 and 8 of family 5's 186, and on none of the others.
  On those 82 the cut to the adstock does not bind: each channel's adstock ranges over the whole
  of its observed range inside the export's box. The fallback, the observed range where the cut
  box holds no plan on the budget, fired on none of the screen's 1730 worlds. Averaged over the
  channels and the worlds, those 82 counted, the box's width was 0.74 (family 12) to 0.97
  (family 2) of the observed range's.
- *The cost.* The laptop was loaded by other work, so these numbers are an order of magnitude.
  The variant's plan, its regression and its water-filling, took 0.22-0.24 CPU-s a world at the
  median and 0.52 at most over the 3155 worlds read. Its record, the arm's shared linear fit
  counted, took 0.38-0.41 and 0.87.
- *Two defects the variant keeps, inherited from `fit_margins` and not fixed here.* (1) The ridge
  prior's precision on each column is a share of the column's own whitened sum of squares, the
  trend's basis included. The trend carries the sales' level, so a channel column may absorb part
  of the level, the more the larger its mean over its variation: a saturated column, or a smoother
  adstock, meets a smaller penalty. The evidence over the retentions, and over a free
  half-saturation, leans that way. On near-noiseless synthetic histories made from family 0's spend
  and linear in the regression's own adstock, the evidence over a common half-saturation read an
  elasticity of 0.88 where the truth's was 1. On those histories it put the retentions at 0.1 and
  0.7 where the truth's were 0.5. On family 0's worlds it read a median elasticity of 0.47 where
  the truth's is 0.965. (2) On a 52-week history, the promoted phases that fall inside the burn-in
  leave their dummies identically zero on the weeks read. The ridge's floor keeps the fit regular
  (family 7's 52-week worlds have three such columns), but the columns carry nothing and are
  counted in the residual degrees of freedom.
- *What it cannot read.* As the margin arm: a geo export and its market are refused, and an export
  without the promotion or the price fits without them. An export whose box, cut to its history's
  range, holds no plan fails the variant (`ObservedRangeError`).
- *The tests* (`tests/test_margin_mmm.py`, `tests/test_margin_plan.py`,
  `tests/scorecard/test_margin_arm.py`). At an infinite half-saturation the curved regression is
  the linear one, every combination's weight to the bit; in `margin ranged`'s box the variant's
  plan and claim are `margin ranged`'s. Each curve's slope is 1 at the status quo and falls with
  the spend, and its rise is its integral. Water-filling meets its KKT conditions on random
  Michaelis-Menten curves, where each free channel spends `sqrt(beta K / lam) - K`, and on random
  mixtures of them. It holds a channel at a bound where its margin outruns or trails the price,
  holds a channel of margin at most 0 at its floor, and gives the rest by the corner where the
  positive channels' ceilings leave budget over; on lines it plans the corner. The adstock range
  and its fallback are tested, and so are the variant's record, its claim and flags, a degenerate
  curved fit failing it alone, and no positive margin playing the status quo. The closed forms the
  tests assert were checked in Maxima: the plan at a price, the elasticity `K / (K + s)` and its
  inverse `K = s e / (1 - e)`, the curve's slope and its linear limit.

## Alternatives considered

- **Reading the half-saturation off each history**, by the evidence over a grid of it with a
  flat prior on the nuisance (the screen's `kappa` by the evidence). One history does not identify
  it (Context), and on the screen it missed the 0.03 bound on family 13 (D13 +0.0286). Freeing it
  is the place to start once the defects above are fixed.
- **The linear margins planned on the curves** missed the 0.03 bound on family 13 (D13 +0.0186).
  **The curved margins at `kappa` 16 planned at the corner** missed it on both families (D12
  +0.0297, D13 +0.0244). At `kappa` 2 and 4 the corner met all four bounds and lay within 0.005 of
  the best min(D12, D13); the tie-break took the curved plan, whose family-6 cell was lower
  (-0.0576 against -0.0406 and -0.0506).
- **`margin ranged`'s box without the adstock cut.** At `kappa` 16 it met all four bounds, but its
  min(D12, D13), +0.0367, fell more than 0.005 below the best, +0.0428; it read family 6 at -0.0202
  against the frozen rule's -0.0576.
