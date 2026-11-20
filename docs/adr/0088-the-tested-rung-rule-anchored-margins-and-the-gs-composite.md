# 0088. The tested-rung rule: margins anchored on the lift tests, and the composite `gs`

Status: proposed, 2026-11-17; adds `mixdyn.calibration.anchors`, the scorecard's `anchored` arm and
its composites (`scorecard.composite`), the composite `gs` among them; changes no default and no
other arm's plan.

## Context

At the tested rung each world's export holds lift tests: go-dark tests, each a channel's weekly
spend `x` taken off for 4 weeks, and the KPI gap `delta_y` it left over those weeks and 8 weeks of
cooldown, with its standard error `sigma`. The margin rule (ADR 0086) reads none of them. A screen
of tested-rung plan rules on M2's development set, M2's scored worlds, found a rule, GS(1.645),
with an implementation written apart from this package, in numpy and scipy:

1. Each test reads a return per euro `rho = -delta_y / x`, of standard error `sigma / x`; each
   channel's return `r_c` is its tests' precision-weighted mean, of standard error `se_c`.
2. A world is routed to `stacked`, the engine's static Bayesian plan, where some channel's tests
   read a return that rises with the spend, the slope statistic of their weighted least-squares
   slope on `ln(x / s_c)` above 1.645 (`s_c` the channel's status quo), or where the export's
   kernel runs longer than the tests' readout, 12 weeks. A rising return is the convex part of an
   S-curve, where a straight line plans badly; a test reads none of the carryover past its
   readout, so under a longer kernel its return reads low. Both are read off the export.
3. Elsewhere the margin regression's margins `b`, of covariance `V` (ADR 0086), are anchored on
   the tests with the regression's bias as a random effect: `d = b - r`,
   `tau_b^2 = max(0, mean_c(d_c^2 - V_cc - se_c^2))`, and the margins' posterior is the
   combination of `b` under `V + tau_b^2 I` and `r` under `diag(se^2)`. It is shrunk by the margin
   rule's empirical Bayes and planned at its corner. A regression that raises leaves the tests
   alone, `r` under `diag(se^2)`.

The bias term is there because the two disagree where the history is confounded. The Hausman
statistic of the regression against the tests, on 3 degrees of freedom, had a median of 1.7, 2.0,
2.0 and 2.5 on families 0, 3, 5 and 7 (the chi-squared's median is 2.37) and flagged 3-5 % of their
worlds at 5 %, but 19 % of family 2's, the family a sentiment confounds, its median 4.2.

A rule's score is the largest mean paired difference of regret a euro, its plan's less a rival's,
over families 0, 2, 3, 5 and 7 and the rivals status quo, equal split, Meridian, PyMC-Marketing and
Orbit. The S-curve guard reads the rule's mean regret less `stacked`'s on families 4 (Hill, Weibull
and logistic curves) and 14 (a 156-week kernel), each at most +0.02. Measured on M2's development
set at the tested rung:

| tested rung | fit split | check split |
|---|---|---|
| GS(1.645): score | +0.0214 (family 2, PyMC-Marketing) | +0.0214 (family 3, PyMC-Marketing) |
| `stacked`: score | +0.0486 (family 7, PyMC-Marketing) | +0.0414 (family 7, PyMC-Marketing) |
| GS less `stacked`, families 4 / 14 | -0.019 / 0 | -0.027 / 0 |
| routed: families 0, 2, 3, 5, 7 / 4 / 14 | 0.11-0.20 / 0.61 / 1 | 0.11-0.15 / 0.57 / 1.00 |

The rule was chosen on the fit split, post hoc, among six gated rules at the threshold named before
they ran, and then read once on the check split, frozen, on its 1036 worlds of families 0, 2, 3, 5,
7, 4 and 14, where no regression raised. Its declared decision held: the score below `stacked`'s by
more than 0.005, each guard at most +0.02, and the status quo beaten on every one of families 0, 2,
3, 5 and 7. By family, on the check split:

| family (worlds) | GS less PyMC-Marketing [95 %] | less Meridian | less `stacked` | routed |
|---|---|---|---|---|
| 0 (174) | +0.011 [+0.001, +0.022] | -0.067 | -0.012 | 0.11 |
| 2 (177) | +0.021 [+0.009, +0.034] | -0.050 | -0.017 | 0.11 |
| 3 (168) | +0.021 [+0.008, +0.037] | -0.069 | -0.013 | 0.15 |
| 5 (156) | +0.008 [-0.003, +0.019] | -0.089 | -0.019 | 0.14 |
| 7 (147) | -0.005 [-0.028, +0.018] | -0.045 | -0.046 | 0.14 |
| 4 (120) | -0.140 | -0.197 | -0.027 | 0.57 |
| 14 (94) | -0.131 | -0.296 | 0 | 1.00 |

Every upper bound against PyMC-Marketing is below 0.05, which `stacked`'s is not on family 7
(+0.059). The routes cost something: on the fit split 11-20 % of the concave families' worlds were
routed by a slope read by chance, about +0.005 to +0.01 on the score.

## Decision

- **`mixdyn.calibration.anchors.read_tests(channels, status_quo, channel, x, delta_y, sigma)`**
  reads the lift rows, one a test, as step 1 states: each channel's return per euro, its standard
  error, its slope statistic and its count of tests. A channel whose tests span less than `1e-9` in
  `ln(x / s_c)` (`FLAT`), tests at one spend, reads a slope statistic of 0. Rows of unequal
  lengths, a row of a channel not planned, a channel without a row, a spend, a standard error or a
  status quo that is not positive and finite, and a gap that is not finite are refused.
- **`mixdyn.calibration.anchors.anchor(mean, cov, roi, roi_se)`** is step 3's combination:
  `tau_b^2` by moments, `P = (V + tau_b^2 I)^-1 + diag(1/se^2)`, the mean
  `P^-1 ((V + tau_b^2 I)^-1 b + diag(1/se^2) r)` and the covariance `P^-1`. It equals the
  generalised least-squares form `r + S (S + V + tau_b^2 I)^-1 (b - r)`, of covariance
  `S - S (S + V + tau_b^2 I)^-1 S`, `S = diag(se^2)`, which the tests check.
- **The scorecard's `anchored` arm** (`scorecard.arms.anchored.Anchored`) reads the tests
  (`tests = True`), runs the margin regression, anchors, shrinks and plans the corner on the
  status quo's weekly total, as the margin arm does. It writes the route among its flags,
  `routed`, with its reasons, `routed by slope` and `routed by kernel`. Its own plan is the
  anchored corner in every world, routed or not: read alone, it is the rule without its route.
  Its flags hold each channel's regression margin, its tests' return and slope statistic, its
  anchored and its shrunk margin, each with its standard deviation, and its retention; `tau_b`, the
  largest slope statistic, the lift rows read, and whether the regression raised.
- **The readout** is `causaldyn_bench.lift_calibration.TEST + COOLDOWN`, 4 + 8 = 12 weeks: the
  bench's test design, which every export states as `lift_weeks` and `lift_cooldown`. The
  observation an arm reads does not carry those two fields, so the arm takes the constants, and its
  settings name them.
- **At k = 0** an export holds no lift row and the rule is not defined there. CHOICE: the arm plans
  the regression alone, the margin arm's plan, routes nothing, and says so under `n/a`; a
  regression that raises is the fit's failure, since no test is left to plan with.
- **CHOICE: an export whose tests cover some channels and not others is refused**, a fit's
  failure: the rule was found on worlds that test every channel. An export without a control
  fits without it, as the margin arm does.
- **CHOICE: the route plays `stacked`'s own record, through a composite, not a fit of its own.**
  `scorecard.composite.compose` adds to a world that scores both `anchored` and `stacked` the
  composite `gs`: the score of `stacked` where the arm routes and of `anchored` elsewhere, with a
  `composite` block that names the arm played, and its cost, the arm's in every world and
  `stacked`'s where it is played, as the procedure would spend them. A router whose fit failed
  routes nothing: `gs` fails with it and plays the status quo, as the bench scores every failed
  fit. A world that does not hold both arms holds no `gs`. The composite is assembled wherever the
  scorecard joins a world's arms, on every read: a run with its joined root
  (`scorecard.claim.read_run`) and with the roots beside it (`scorecard.claim.joined_runs`), and a
  pilot's arms with the stored run's (`scorecard.development.joined`). Assembling it twice gives
  it once.
- **A run reads `gs`** by declaring it its default above 0 on a run that plans `anchored` and
  `bayes`, or with `anchored` in a root beside the run, read by `claim read` with it, or from the
  run's joined root where that root plans both arms: `scorecard.claim.read_run` joins the
  router's records and the routed arm's (`scorecard.composite.parts`), so a run whose main root
  holds every rung fits `stacked` at the tested rung alone, in its `-ts` root. A joined root that
  plans one of the two is refused. The selection reads it as a candidate named
  `gs` at the tested rung, its CPU-seconds the composite's cost. No record may take a composite's
  name (`scorecard.run.names`).
- **Defaults are unchanged.** No run before this one plans `anchored`, so no record gains `gs`; and
  the claim's rivals are the built-in arms and the run's tools, so `gs` is no comparison's rival.

## Consequences

- *Parity.* On eight of M2's worlds at the tested rung in the development set's fit split, of
  families 0, 3, 5 and 7, two of family 2 (one routed, one whose bias reads above 0) and two of
  family 4 (one routed), the route, each channel's slope statistic, its return and its standard
  error, the regression's margins, `tau_b`, the anchored mean and covariance and the shrunk means
  are the research implementation's to 1e-9 relative, and the plans equal; so are the tests alone
  shrunk and planned. Scored by the bench's scorer beside M2's recorded `stacked` record, the
  composite's regret is the rule's on each of the eight, within 1e-9 a euro
  (`tests/fixtures/gs_worlds.json`, its expected values written by that implementation).
- *The screen* (M2's development set, fit split, tested rung, the archived exports, no rival
  refitted; the question and the rule written before it ran). On 55 worlds drawn by a seed, 8 of
  each of families 0, 2, 3, 5 and 7, 10 of family 4 and 5 of family 14, `gs` was read twice:
  scored with M2's `stacked` record in one world record, and through the selection's reading of a
  pilot root that held the arm's records beside M2's stored roots. In both its regret equalled the
  rule's on every world, to the bit; the arm's route and largest slope statistic equalled the
  research implementation's on all 55, 18 of them routed, 5 by the kernel (family 14), 13 by the
  slope alone. No fit failed, and the scorer moved no plan.
- *What the route buys*, read on the screen's routed worlds alone, exploratory: on family 4's 8 the
  arm's own plan scored 0.517 a euro and `stacked`'s 0.363, on family 14's 5, 0.520 and 0.425; on
  the 5 routed worlds of families 0, 3, 5 and 7, `stacked` lost to the arm's own plan on 3. Worlds
  this few are no claim.
- *The cost.* The arm adds a regression to the margin arm's: on a laptop loaded by other work, so
  to be read as an order of magnitude, 0.04-0.10 CPU-s a world. M2's records state `stacked`'s fit
  and plan at 170-500 CPU-s on the same worlds. A run that records `stacked` in every world pays
  nothing more for the route; the composite's cost counts `stacked`'s where it is played.
- *What it cannot read.* A geo export's market and an export whose tests leave a channel out are
  refused; an export without a control fits without it; an export without a lift row, at any
  rung, is planned as at k = 0.
- *The tests* (`tests/test_anchors.py`, `tests/scorecard/test_anchored_arm.py`): the parity above;
  the anchored posterior against its generalised least-squares form at three values of `tau_b`;
  tests whose return rises with the spend reading a positive slope, falling ones a negative slope,
  and tests at one spend none; the inputs refused; the arm routed by a rising return, by a kernel
  of 13 weeks and not 12, and not by tests at one spend; a regression that raises leaving the tests
  alone; the k = 0 plan; the composite's plan, failure and cost; and the claim's and the
  selection's readers assembling it, as the default above 0 and beside the run.

## Alternatives considered

- **An arm that fits `stacked`'s engine itself where it routes**: a sampler run of 170-500 CPU-s in
  each routed world, more than half of family 4's and all of family 14's, to remake a plan M3
  records anyway, and a second plan of the same model under another seed.
- **Other combinations and gates**, measured on the fit split at the tested rung: the regression
  and the tests combined without the bias term, gated at 1.645 (+0.0271); a Hausman pretest that
  plans the tests alone where the two disagree, gated at 1.645 (+0.0232); the three gated at 2
  (+0.0257, +0.0213, +0.0200); and the bias-aware rule without the gate (+0.0169), which failed the
  S-curve guard: +0.057 on family 4, no fit on family 14.
- **Reading the readout off the export.** The exports state the test design, the observation the
  arms read does not; carrying it would change the bench's observation for one constant that is
  the same in every export.
