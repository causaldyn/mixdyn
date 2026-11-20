# 0089. The curved tested-rung rule: a curve posterior off the tests, and the composite `gs curved`

Status: proposed, 2026-11-17; adds `mixdyn.calibration.anchors.read_curves`,
`mixdyn.models.margin_mmm.project` and `mixdyn.decisions.margin_plan.water_fill_tabulated`, the
`anchored` arm's variant `anchored curved` and the composite `gs curved`; changes no default and no
other variant's plan.

## Context

The tested-rung rule GS(1.645) (ADR 0088) plans the corner that a straight line makes of each
channel's anchored margin, and routes a world to `stacked` where the tests read a convex curve or
the kernel outruns their readout. Its weak cells were the concave worlds. On the M3 pilot's 30 + 30
worlds of families 12 (Meridian's generator) and 13 (PyMC-Marketing's) at k = 4, its mean regret
was 0.0642 and 0.1560 a euro, against `stacked`'s 0.0473 and 0.0891 and PyMC-Marketing's 0.0374 and
0.0800; its worst mean less a rival was +0.0268 and +0.0760 (PyMC-Marketing both).

Three readings, written apart from this package in numpy and scipy, fixed the design. They read
70 + 70 worlds of families 12 and 13 outside the pilot's, and the truth only as an evaluator:

- **The inputs.** On family 12 the regression's margin is noise against the truth's (the 10-90 %
  range of their ratio on one channel ran from -1.6 to 13). So a statistic of the regression
  against the tests does not see curvature: a pooled one-sided linearity statistic rejected in 7 %
  and 0 % of the two families' worlds at 1.645. The tests see it: the least slope statistic lay
  below -1.645 in 49 % and 34 %.
- **Where the regret comes from.** The corner on the truth's own margins at the status quo
  regretted 0.033 a euro on each family, against GS's own plan's 0.066 and 0.200. So on family 13
  most of the loss is the margins' reading, not the corner. A test of a bursty channel, far above
  the status quo, reads a chord deep in the saturation: its return over the true margin had a
  median of 0.76 on family 13.
- **Which curve.** Through the truth's own margin and elasticity, a Michaelis-Menten curve planned
  family 13 at 0.0035 and a logistic at 0.0000; on family 12 the reverse, 0.0001 against 0.0160.
  One family of curves does not serve both generators.

The screen read 47 variants of GS, each declared before its regrets were computed, on M2's fit
split at the tested rung (families 0, 2, 3, 5 and 7 for the score, 4 and 14 for the S-curve guard)
and on the 70 + 70 worlds, with `stacked` fitted on the first 15 of each family as a calibration.
The variants: further routes (by the linearity statistic, by a falling slope statistic); GS's
margins water-filled on Michaelis-Menten curves of one fixed elasticity; a posterior over the
elasticity for each channel, with Michaelis-Menten curves alone or with logistic ones beside
them, at a prior mass on the line from 0.5 to 0.8, or estimated a world, or with the curve shared
by a world's channels; the regression's bias re-read from that posterior; and the tests near the
status quo alone.

A rule declared before the screen froze one variant. A variant was eligible with (E1) a fit-split
score at most GS's +0.0214 + 0.005, (E2) on families 4 and 14 a mean regret at most `stacked`'s
+ 0.02, and (E3) on each of families 12 and 13 a composite's mean over the 15 calibration worlds at
most `stacked`'s + 0.01. Among the eligible the lowest score won, scores within 0.003 tied, ties
going to the smaller largest calibration excess, then to the fewer tuned constants. Sixteen
variants were eligible. Six tied at the lowest score:

| fit split and the calibration worlds, tested rung | score | guard, family 4 | largest calibration excess | eligible |
|---|---|---|---|---|
| GS(1.645) | +0.0214 | -0.019 | +0.1113 | no (E3) |
| Michaelis-Menten, shared by the channels, line prior 0.5 | +0.0147 | -0.002 | +0.0015 | yes |
| Michaelis-Menten, line prior 0.6 | +0.0151 | +0.016 | +0.0015 | yes |
| both families, line prior 0.7 (frozen) | +0.0156 | +0.010 | +0.0014 | yes |
| both families, line prior 0.7, the bias re-read | +0.0156 | +0.010 | +0.0014 | yes |
| both families, shared by the channels, line prior 0.5 | +0.0173 | -0.002 | +0.0016 | yes |
| both families, line prior 0.8 | +0.0134 | -0.001 | +0.0299 | no (E3) |
| both families, line prior 0.5 | +0.0220 | +0.027 | -0.0005 | no (E2) |
| fixed elasticity 0.95 | +0.0150 | -0.007 | +0.0753 | no (E3) |

The smallest largest excess was +0.0014, of the two variants with both families at a line prior
of 0.7; the one without the re-read bias has one constant fewer, and the rule froze it: GS-ML(0.7).
The first of the six is as defensible: its excess is 0.0001 larger.

Then the frozen rule was read once, on M2's check split at the tested rung (1036 worlds) and on
the pilot's 30 + 30 worlds at k = 4, the composite playing the recorded `stacked` where it routes.
Its declared decision held: the check-split score at most +0.0264, each guard at most +0.02, and on
each of families 12 and 13 a worst mean less a rival at most `stacked`'s own + 0.01.

| check split, tested rung (worlds) | the frozen rule less PyMC-Marketing [95 %] | less Meridian | less `stacked` | GS less PyMC-Marketing | routed |
|---|---|---|---|---|---|
| family 0 (174) | +0.0047 [-0.002, +0.012] | -0.0734 | -0.0181 | +0.011 | 0.11 |
| family 2 (177) | +0.0061 [-0.004, +0.016] | -0.0645 | -0.0316 | +0.021 | 0.11 |
| family 3 (168) | +0.0165 [+0.005, +0.030] | -0.0738 | -0.0182 | +0.021 | 0.15 |
| family 5 (156) | +0.0033 [-0.004, +0.011] | -0.0932 | -0.0233 | +0.008 | 0.14 |
| family 7 (147) | -0.0059 [-0.026, +0.013] | -0.0461 | -0.0474 | -0.005 | 0.14 |
| family 4 (120) | -0.1067 | -0.1638 | +0.0069 | -0.140 | 0.57 |
| family 14 (94) | -0.1313 | -0.2961 | 0 | -0.131 | 1.00 |

The score was +0.0165 (family 3, PyMC-Marketing), against GS's +0.0214 and `stacked`'s +0.0414;
the guards +0.0069 and 0.

| the pilot, k = 4 | the frozen rule's mean regret | its worst mean less a rival | GS's | `stacked`'s | declared bound |
|---|---|---|---|---|---|
| family 12 (30 worlds, 3 routed) | 0.0396 | +0.0022 (PyMC-Marketing) | +0.0268 | +0.0099 | +0.0199 |
| family 13 (30 worlds, 5 routed) | 0.0752 | -0.0048 (PyMC-Marketing) | +0.0760 | +0.0091 | +0.0191 |

The curve costs on S-curves the route does not catch. On family 4's check-split worlds that were
not routed, 52 of 120, the rule regretted 0.077 a euro a world more than GS's corner: its guard
moved from GS's -0.027 to +0.0069, inside the +0.02 it was held to. The M3 selection's predictive
(`scorecard.development.predictive`), read on the check split and the pilot's worlds, was 1.000,
1.000, 0.945, 1.000 and 0.997 on families 0, 2, 3, 5 and 7, and 1.000 and 0.998 on families 12 and
13. Several frozen rules have read the check split and the pilot's worlds, the later ones written
after the earlier reads. So these predictives are optimistic, and M3's fresh worlds are the test.

## Decision

- **The rule, GS-ML(0.7).** Its routes are GS's, unchanged. Elsewhere it takes GS's regression
  margins `b` of covariance `V`, the tests, the bias `tau_b^2`, the anchored margins and their
  shrunk means `m` (ADR 0088). Each channel then carries a posterior over 83 curve columns
  through its margin at the status quo `s`: Michaelis-Menten curves `beta a / (K + a)` of the
  adstock `a`, `K = s e / (1 - e)`; logistic curves `beta tanh(k a)`, `2 k s / sinh(2 k s) = e`;
  each at the 41 elasticities at the status quo `e` of 0.05 to 0.95 by 0.025 and 0.97, 0.98, 0.99
  and 0.995; and the straight line. The prior puts 0.7 on the line and splits 0.3 evenly by the
  two families and uniformly over the elasticities. Given a column, one weighted least squares
  reads the margin at `s` off two kinds of reading, each the margin times a factor the column
  fixes, over the line's factor. The regression's margin `b_c`, of variance `V_cc + tau_b^2`, has
  the factor `B`: the coefficient the regression reads off the column's curve of the channel's
  adstock, per unit of the curve's margin at `s`. Each test's return per euro, of variance its
  standard error squared, has the factor `R_j`: the exact lift the curve gives the test's 4 dark
  weeks over its 12-week readout, over the dark spend, per unit of the curve's margin at `s`. A
  column's weight is its prior times `exp(-rss / 2) / sqrt(information)`, its marginal likelihood
  under a flat prior on the margin. The channel's margin curve is `m_c` times the posterior
  mixture of the columns' shapes, `((K + s) / (K + p))^2`, `(cosh(k s) / cosh(k p))^2` and 1, each
  column weighed further by its least-squares margin over the line's, held at 0 or above. A
  channel whose `m_c` is not positive keeps a flat margin. The plan water-fills these curves in the
  box at the status quo's weekly total.
- **`mixdyn.models.margin_mmm.project(sales, spend, promotion, price, kernel, retention)`** reads
  the regression at one retention combination as a linear map (`Projection`): the AR(1)
  coefficient that the sales give at that combination, its whitening, and the ridge's map
  `(X' X + Lambda)^-1 X'`. `Projection.slope(channel, series)` is the coefficient the regression
  reads on a channel's column with `series` in the sales' place. The rule reads it at the
  combination of the most evidence (`MarginFit.top`).
- **`mixdyn.calibration.anchors.read_curves(...)`** is the posterior above (`Curves`): each
  channel's half-saturations and logistic rates, its posterior weights over the columns and each
  column's margin ratio. `Curves.margin(channel, level)` is the channel's margin curve,
  `Curves.rise` its integral from the status quo, and `Curves.elasticity` its posterior mean
  elasticity. `logistic_rate(e)` solves `2 u / sinh(2 u) = e` by 200 bisections. `anchor`'s checks
  and its moment for `tau_b^2` moved into private helpers that `read_curves` shares; `anchor`
  returns what it returned.
- **`mixdyn.decisions.margin_plan.water_fill_tabulated(curves, lower, upper, budget)`** is the
  rule's planner. It tabulates each channel's margin at 4001 spends across its box and holds it
  falling by a running minimum. At a price, each channel spends the most tabulated spend whose
  margin is at least the price. The price is bisected 200 times. What the plan at the upper price
  leaves of the budget is shared by the channels the lower price moves, in proportion to their
  moves.
- **CHOICE: two water-filling planners, because parity forces the second.** `water_fill`
  (ADR 0087) is the exact one: it bisects each channel's spend at a price along any falling slope,
  a mixture among them, and the Michaelis-Menten closed form is its one-component case. The frozen
  tested-rung rule was read with the tabulated planner, whose plan sits on a table of steps of
  1/4000 of each box. On the worlds the rule was read on, `water_fill` on the same curves moved the
  regret by up to 0.0014 a euro (below 1e-15 at the median, above 1e-9 on 420 of the 1096
  worlds, in either direction as often), against the 1e-9 the port is held to. So
  `anchored curved` plans with the tabulated planner, and its plans are the frozen rule's. The
  tests hold the tabulated plan within its table's steps of the exact one.
- **The variant `anchored curved`** (`scorecard.arms.anchored`) shares the arm's fit, its route
  and its flags. It reads the curves in its plan, so their cost is its own. Where the export holds
  no lift row, where the regression raised, or where no margin is positive, no curve is read: the
  plan is `anchored`'s. Its record claims the plan's gain along the curves,
  `planned sum_c m_c (H_c(p_c) - H_c(s_c))`, `H_c` the integral of the channel's margin over
  `m_c`, with the normal interval of the shrunk margins' covariance at 90 %; with the line's prior
  at 1 it is `anchored`'s claim. Its flags add each channel's posterior weight on the line and its
  posterior mean elasticity. Its settings add the line's prior, the elasticities, the curve
  families, the table's points and the price's bisections.
- **The composite `gs curved`** (`scorecard.composite.GS_CURVED`) plays `stacked` where
  `anchored curved`'s record says the world is routed, and `anchored curved` elsewhere. It is
  assembled wherever `gs` is: `scorecard.claim.read_run`, `scorecard.claim.joined_runs` and
  `scorecard.development.joined`. A run reads it as `gs` is read, by the name `gs curved`. No
  record may take its name.
- **Defaults are unchanged.** `anchored`'s plan, record and route are what they were. A root that
  declares the `anchored` arm now holds both variants' records.

## Consequences

- *Parity.* The variant's records were made by the arm's harness from the archived exports. The
  bench's `score_world` scored them against each world's archived export beside the recorded
  `stacked` record, and `compose` assembled `gs curved`. Its regret is the frozen rule's on every
  world the rule was read on, with no difference in the last bit: on M2's 1036 check-split worlds
  and on the pilot's 60. With the line's prior set to 0.8 it is the research rule's at that prior
  on M2's 1339 fit-split worlds, to the last bit too. The route agreed on every world, `gs`
  assembled in the same call was GS's to the last bit, no record failed and the scorer moved no
  plan. On eight fit-split worlds
  (`tests/fixtures/gs_curved_worlds.json`, its expected values written by the research
  implementation) the plan, each channel's weight on the line and the regret are that
  implementation's.
- *The departures, none of which moved a regret past those differences.* (1) The rule as found
  fitted no regression under a kernel past the readout and planned the tests alone there; the arm
  fits the regression under every kernel, as `anchored` does. Every such world is routed, so the
  composite plays `stacked` either way; only the variant's own plan, which no procedure plays
  there, differs. (2) The rule as found took the box and the status quo from the bench's truth;
  the arm takes them from the export, which states the same ones (equal on every world read).
  (3) The rule as found returned the status quo where its table could not spend the budget; here
  a box that holds no plan on the budget is refused, a failed plan. No world read had one.
- *A behaviour the variant keeps, inherited from the rule and not changed here.* The tests'
  adstock reads the history before its first week as no spend, where the regression imputes it. A
  test whose readout reaches the history's first `kernel - 1` weeks so reads its control low. On
  the worlds read no test starts before week 20, and no kernel planned on exceeds 12 weeks. The
  line's factor for a test, the share of its dark spend's carryover that the readout holds, is 1
  under a kernel of at most 9 weeks and below 1 up to 12: GS reads the return as the margin there,
  and the curves are read relative to the line as GS reads it.
- *The cost.* The laptop was loaded by other work, so these numbers are an order of magnitude.
  The variant's plan, the curves and the table, took 0.04 CPU-s a world at the median and
  0.19 at most; its record, the arm's shared fit counted, 0.25 and 0.81.
- *What it cannot read.* As `anchored`: a geo export's market and an export whose tests leave a
  channel out are refused, and an export without a control fits without it. A test whose dark
  weeks do not lie in the history fails the variant's plan alone.
- *The tests* (`tests/test_anchors.py`, `tests/test_margin_plan.py`,
  `tests/scorecard/test_anchored_arm.py`). A logistic column's rate and a Michaelis-Menten
  column's half-saturation read their elasticity. Readings that one column makes exactly put the
  posterior on that column and its shape on the margin curve. With the line's prior at 1 the curve
  is a line and the variant's plan and claim are `anchored`'s, on the eight worlds; with it at 0
  the line weighs nothing. A curve's rise is the integral of its margin. The tabulated planner
  spends the budget in the box within its table's steps of the exact plan, on Michaelis-Menten
  curves and on mixtures with a logistic and a line, whose exact plan meets its KKT conditions; on
  margins that never fall it is the corner; a margin at most 0 stays at its floor and a margin
  that outruns the others at its ceiling. The parity above on the eight worlds; the record, its
  flags and settings; a regression that raises and an export without tests planned as `anchored`
  plans them; the composite, and its name refused to a record. The closed forms the tests assert
  were checked in Maxima: the two elasticities and `K = s e / (1 - e)`, the two shapes, the two
  rises and their limits at the line, and the column's weight as the flat-prior integral of its
  likelihood.

## Alternatives considered

- **Routing more worlds to `stacked`**, by a falling slope statistic or by the linearity
  statistic. The linearity statistic changed nothing (score +0.0214, calibration excess +0.1113).
  The falling slope at 1.645 routed 28-43 % of M2's linear worlds and missed the score's bound
  (+0.0325); at 2.326 it kept the bound (+0.0258) and missed the calibration (+0.0501).
- **One fixed elasticity for every channel.** At 0.95 it scored +0.0150 and missed the calibration
  (+0.0753); at 0.9 and below it missed the score too. Unlike the k = 0 rule (ADR 0087), the
  tested rung has tests that read each channel's curvature, so the curvature need not be a
  constant.
- **Michaelis-Menten columns alone.** Eligible at a line prior of 0.6 and tied with the frozen
  rule; the logistic columns were added because a Michaelis-Menten curve saturates too slowly for
  family 13's truth (Context).
- **The curve shared by a world's channels, and the line's prior estimated a world.** The shared
  curve at a prior of 0.5 tied with the frozen rule and lost the tie-break by 0.0001. The
  estimated prior missed both the score and the calibration (+0.0289, +0.0700).
- **Re-reading the regression's bias from the curve posterior.** GS's moment counts the gap that
  curvature makes between a margin and a chord as confounding. The re-read variant scored the same
  to four decimals and has one constant more.
- **The exact planner for this rule.** It is the exact plan on the same curves, and it would
  leave one planner; on the worlds read it regretted less as often as more. It is not the rule
  that was frozen and read; the swap is the place to start when the rule is next revised on fresh
  worlds.
