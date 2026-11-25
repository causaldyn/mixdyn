# 0090. A closed-form model over geos fitted by empirical Bayes, and its plan of a mixture of curves

Status: proposed, 2026-11-17; adds `mixdyn.models.geo_eb`, `mixdyn.decisions.geo_plan.plan_mixture`
and the scorecard's `geo-eb` arm, its one variant `geo eb`; changes no default, no other arm's plan
and no pre-registration. Whether a run claims the arm, and whether it then drops the `bayes-geo`
fit, is a call this record does not make.

## Context

ADR 0083's rule picked `bayes geo plug-in` as the claim's arm on the geo families. Its fit is four
Markov chains of the model over geos: 1 725, 3 095 and 3 660 CPU s a world-rung at three, eight and
ten geos, the mean of `cost.fit_cpu_seconds` over M2's `bayes geo plug-in` records, which is the
fit alone; a whole process took 1 938, 3 898 and 5 460 to 5 717 CPU s. It was about a quarter of
the next run's CPU hours as designed.

A screen on the development set (ADR 0081: M2's scored worlds of family 8 and of family 10 at three
and eight geos, the fit split alone, and seventy laptop worlds an environment of family 9) asked
whether a rule that reads the export alone, in at most 30 CPU s a world, scores within 0.01 of
`bayes geo plug-in` by ADR 0083's score. The question, the data, fifteen variants, the score and
the selection rule were written before any variant's numbers. The rule this record ports is the
one that screen froze, before the held-out worlds were read once.

## Decision

### The model (`mixdyn.models.fit_pooled_geos`, named `fit_geo_eb` until 0.3.0)

The module's docstring states it in full. In short, from a geo panel, its lift estimates and the
kernel's length:

- **Columns.** Per channel and grid point (retention `r`, curvature `kappa`): the normalised
  geometric adstock of the given length, nothing spent before the first week, the first
  `length - 1` weeks not read; spend per head relative to the channel's national mean,
  `u = a_gc / (p_g xbar_c)`; the column `p_g xbar_c F(u)`, `F` Michaelis and Menten's curve scaled
  to a margin of 1 at `u = 1`, the line at `kappa = inf`. A coefficient is the margin at the
  national mean spend per head. The grid is 7 retentions (0 to 0.9 by 0.15) by 9 curvatures
  (0.25 to 32 by doubling, and the line).
- **Nuisance.** Each geo its own intercept, 2 yearly harmonics and a linear trend, partialled out
  of the KPI and of every column geo by geo.
- **Noise.** Each geo's variance from a reference fit (every channel at retention 0.45 on the
  line), held at every grid point.
- **Pooling.** `b_gc = mu_c + delta_gc`, `delta_gc ~ N(0, (0.4 m_c)^2)`, `m_c` from the reference
  fit's coefficients; `mu_c` under a wide normal prior.
- **Tests.** Each lift estimate one Gaussian row: the change in its geos' summed contribution over
  its readout, linear in their coefficients given the grid point.
- **Fit.** The marginal likelihood of a choice of grid points is closed; 3 sweeps of coordinate
  ascent over the channels; then each channel's posterior over its 63 points, the others at
  theirs, weights above `1e-4` kept, and the coefficients' posterior mean under each.

The identities are checked in Maxima (`validation/geo_eb.mac`, 15 verdicts): Frisch, Waugh and
Lovell; the least-squares split of the quadratic form; the Gaussian integral; the closed form
assembled against the brute-force Gaussian of two geos and a test row, exactly; the scaled curve;
a national channel as one cell; a cell's slope.

### The plan (`mixdyn.decisions.geo_plan.plan_mixture`)

A cell's expected worth is the mixture, over its channel's kept points, of its worth at the
posterior-mean coefficient: the quarter and the kernel's tail, the history's carryover in. A
channel bought nationally is one cell of its national spend, its scale the national mean and its
coefficient `sum_g p_g b_g`. The planner reads each cell's slope at 401 spends evenly over its box,
makes it non-increasing, and water-fills: one price, bisected 200 times to the budget, and the two
plans that bracket the budget joined at it.

### The arm (`scorecard/arms/geo_eb.py`, declared as `geo-eb`)

It reads the export through `scorecard.arms.panel` (`panel`, `estimates`, `cells`), as the
`bayes-geo` arm does, fits once and plans `geo eb`. It claims no gain (below), draws nothing and
has no seed; a smoke run is the run. It carries no forecast and no response the bench reads, as
`bayes geo plug-in`.

### Where the rule's arithmetic and the library's differ, the rule's is kept

- `plan_geos` is not used. It takes one `chc.response.Channel` a cell and solves the plan exactly;
  a cell here is a weighted sum of curves, which no one channel holds. A plan at the single best
  grid point, the one curve a cell `plan_geos` could take, was a variant of the screen and scored
  0.008 worse.
- `plan_mixture` is exact to its grid, not to rounding: on concave cells the cells inside their
  boxes meet one price to about `1e-4` of it (tested). Its numbers are the frozen rule's.
- The curve is `chc.response.MichaelisMenten` rescaled to its margin at 1, held in NumPy: the fit
  reads 63 points a channel in one array and no JAX.
- The adstock is `mixdyn.models.margin_mmm.adstock`, the library's, which equals the rule's own
  loop to rounding.

### Where the port departs from the code the rule was frozen in

None changes the rule; the regrets below are equal to rounding.

- The frozen code read the export's arrays; the port reads `scorecard.arms.panel`'s panel. A
  national channel's spend in a geo is then the panel's, the national spend times the geo's share,
  not the export's own column, which the panel holds to `1e-9` of it. The channels are in the
  panel's order, a geo's own first, and the coordinate ascent visits them in it; on the bench's geo
  families that is the export's order.
- A test is a `LiftEstimate`: its readout, its lift and its standard error are the estimate's, and
  each geo it names spends its history scaled by the group's `spend / planned`. On the bench's
  go-dark rows that is the frozen code's reading: the dark weeks at 0, a readout of the dark weeks
  and the cooldown. The rows enter by channel, not in the export's order.
- A panel that leaves no residual degree of freedom is refused; the frozen code floored the degrees
  of freedom at 1.
- A channel that moves in no geo is refused by name: one whose reference column keeps, in every
  geo, less than `1e-6` of its sum of squares once the geo's nuisance is out, as a channel never
  spent or spent the same every week. The frozen code read its margin off rounding. Over the 640
  worlds the rule was run on, the least share a channel kept in its best geo was 0.018. A geo
  where a channel does not move is not refused: its margin there is the channel's.
- A lift estimate whose readout begins in the kernel's burn-in is refused, and so is a marginal
  likelihood that is not finite, as where the KPI's squares overflow; the frozen code planned the
  status quo there without an error. No export the rule was run on meets either.
- A geo's noise variance is floored at `1e-12` of its KPI's variance once its nuisance is out,
  and a scale has no floor; the frozen code floored both at `1e-12` in the money's unit, which a
  KPI in billionths of the unit met. Neither floor binds on an export the rule was run on. The KPI
  and the spend in another unit now give the same plan in it (tested at `1e-15`, `1e-9` and
  `1e9`), and a geo whose KPI does not move at all is refused.
- The plan's weeks are the export's `planned`, 13 on every family, where the frozen code held 13.
  A budget no plan in the box spends is refused.
- The other fourteen variants' switches are not ported.

### The arm claims no gain

The frozen code computed no gain, and the arm claims none. The gain the fit itself gives is the
plan's expected worth less the status quo's at the posterior means, read on the plan chosen to
maximise it. On the fit split that reading was 1.3 to 2.2 times the gain the plan realised. Per
euro of the budget, the claim, the realised gain and the bench's `uplift`, the claim less the
realised gain, with `bayes geo plug-in`'s recorded `uplift` on the same worlds:

| worlds | rung | n | claimed | realised | claimed less realised (se) | `bayes geo plug-in`'s |
|---|---|---|---|---|---|---|
| family 10, 3 geos | 0 | 40 | 0.1379 | 0.0709 | `+0.0670` (0.0126) | `+0.0250` |
| family 10, 3 geos | 4 | 40 | 0.1409 | 0.0709 | `+0.0701` (0.0111) | `+0.0260` |
| family 10, 8 geos | 0 | 40 | 0.0974 | 0.0645 | `+0.0329` (0.0062) | `+0.0252` |
| family 10, 8 geos | 4 | 40 | 0.0954 | 0.0655 | `+0.0298` (0.0062) | `+0.0221` |
| family 10, 50 geos | 0 | 40 | 0.1143 | 0.0530 | `+0.0614` (0.0099) | no record |
| family 10, 50 geos | 4 | 40 | 0.0952 | 0.0553 | `+0.0399` (0.0077) | no record |
| family 8, `policy-0` | 0 | 66 | 0.1007 | 0.0785 | `+0.0222` (0.0037) | `+0.0075` |
| family 8, `policy-0` | 4 | 66 | 0.1030 | 0.0781 | `+0.0249` (0.0038) | `+0.0064` |
| family 8, `policy-0.5` | 0 | 66 | 0.0973 | 0.0657 | `+0.0315` (0.0056) | `+0.0178` |
| family 8, `policy-0.5` | 4 | 66 | 0.0893 | 0.0683 | `+0.0210` (0.0047) | `+0.0165` |

The claim was below 0 in no world; the realised gain was, in up to 6 worlds of a row. On thirty
pilot worlds an environment of family 9 the sign turns: `-0.0013` and `-0.0056` at spillover 0.2,
`-0.0364` and `-0.0249` at 0.5. It is the optimiser's curse and no error of units: on worlds of
the bench's geo generator at 3 and 8 geos, six a cell, with the KPI's noise at a tenth and a
hundredth of the bench's, the claim was 1.05 to 1.11 times the realised gain. The fit keeps each
coefficient's posterior mean under each grid point, not the joint posterior a correction or an
interval needs. So a record holds no `gain` and no `gain_interval`, and its `n/a` says why; the
bench scores the plan's regret alone, and a report prints no claimed-less-realised figure for the
arm.

### The numbers the rule was frozen and read on

Regret per euro, the rule less a rival, mean paired difference; ADR 0083's score is the largest
over families 8 and 10 and the five rivals (status quo, equal split, Meridian, PyMC-Marketing,
Orbit).

**Frozen on the fit split** (132 worlds of family 8, 80 of family 10): score less
`bayes geo plug-in`'s `+0.0018` at rung 0 and `+0.0047` at the tested rung; mean regret 0.0200 and
0.0189 on family 8, 0.0233 and 0.0228 on family 10, against `bayes geo plug-in`'s 0.0135, 0.0127,
0.0215 and 0.0181. The variant with `bayes geo plug-in`'s priors on the grid scored best, `-0.0002`
at its worse rung; the declared tie rule (within 0.005, then fewer CPU seconds, then fewer
settings) chose this one. That variant was never read on held-out worlds.

**Read once on held-out worlds**, under a rule declared before the read:

| worlds | rung | n | rule's mean regret | `bayes geo plug-in`'s | rule's score less its |
|---|---|---|---|---|---|
| family 8, M2's check split | 0 | 108 | 0.0227 | 0.0162 | `+0.0040` (both families) |
| family 8, M2's check split | 4 | 108 | 0.0212 | 0.0152 | `+0.0053` (both families) |
| family 10, 3 and 8 geos, check split | 0 | 80 | 0.0248 | 0.0208 | as above |
| family 10, 3 and 8 geos, check split | 4 | 80 | 0.0231 | 0.0185 | as above |
| family 9, spillover 0.2, a pilot | 0 | 30 | 0.0237 | 0.0285 | |
| family 9, spillover 0.5, a pilot | 0 | 30 | 0.0417 | 0.0635 | |
| family 9, spillover 0.2, a pilot | 4 | 30 | 0.0239 | 0.0295 | |
| family 9, spillover 0.5, a pilot | 4 | 30 | 0.0461 | 0.0787 | |

The declared bounds were 0.01 on the first four rows' score difference and `+0.02` on family 9's
worst mean against a rival; the worst there was `+0.0101`, against PyMC-Marketing at spillover 0.5
and the tested rung, where `bayes geo plug-in`'s was `+0.0428`. Both held. On families 8 and 10 the
rule beat all five rivals in every cell and trailed `bayes geo plug-in` by 0.004 to 0.006 of mean
regret. The claim's predictive (ADR 0081) was 1.000 for both arms in every family and rung, so it
does not tell them apart.

**The port against the frozen code**, the arm run through `scorecard.arms.base.records` and the
bench's `score_world` on each world's archived export: 1 200 world-rungs (the 248 held-out worlds
above, the 212 of the fit split and 140 laptop worlds of family 9, at both rungs), the largest
difference of regret `1.9e-15`, no failed fit. At fifty geos (12 fit-split worlds of family 10,
exploratory): `4.4e-15`, mean regret 0.0335 and 0.0339.

**CPU a world-rung** on a laptop, fit and plan, one thread: 0.07 s at three geos (largest 0.10),
0.14 s at eight (0.20), 0.16 s at ten (0.25); at fifty geos 0.89 s at rung 0 and 2.30 s at the
tested rung (largest 3.16). The machine was shared while they were read, so they are indicative.

## Consequences

- A run may declare `geo-eb` beside or in place of `bayes-geo-plug-in`. A world-rung costs 0.07 to
  0.16 CPU s at three to ten geos, where the Markov chains' fit cost 1 725 to 3 660.
- The fifty-geo rung, which the model over geos could not run in a day (ADR 0083), costs 0.9 CPU s
  at rung 0 and 2.3 at the tested rung.
- The rule is weaker than `bayes geo plug-in` on families 8 and 10 by 0.004 to 0.006 of regret and
  stronger on family 9's spillover worlds. It models no spillover.
- Its tests do not help where they spill: at spillover 0.5, on thirty pilot worlds of family 9, the
  regret at the tested rung less rung 0's was `+0.0108` (se 0.0043); at 0.2 it was `+0.0018`
  (0.0028).
- The coordinate ascent's selection depends on the channels' order: with the order reversed it was
  another in 6 of the fit split's 424 world-rungs.
- The arm claims no gain. A claim calibrated on held-out worlds is work for after M3.
- The rule's constants (the grid, the spread 0.4, the sweeps, the planner's grid) are frozen with
  the read above. Changing one makes another rule, to be screened and read anew.
- The gap to `bayes geo plug-in` on families 8 and 10 lies in the geos' margins: with the true
  margins and the rule's curves the plan's regret was 0.004 to 0.005 on the fit split.

## Alternatives considered

- **`bayes geo plug-in`'s priors on the grid.** The best fit-split score, tied with this rule under
  the declared rule and not read on held-out worlds. Not ported: the read was of this rule.
- **A plan at the single best grid point**, which `plan_geos` could take. Rejected: 0.008 worse on
  the screen's score.
- **The line alone**, a linear model planned at a corner. Rejected: 0.026 worse, and it failed the
  screen's guard on family 9.
- **The spread chosen by type-II maximum likelihood.** Rejected: no better than the fixed 0.4, and
  more settings.
- **A mean-field spillover column.** Rejected: better on family 9 at spillover 0.5, 0.009 worse on
  the score.
- **The frozen code copied in as it was**, reading the export's arrays. Rejected: the arms read an
  export through one reader, and the library's model takes a panel and estimates, as its other
  models do.
