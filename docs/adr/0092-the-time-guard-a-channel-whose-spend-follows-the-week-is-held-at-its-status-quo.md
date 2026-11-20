# 0092. The time guard: a channel whose spend follows the week is held at its status quo

Status: proposed, 2026-11-17; adds `mixdyn.decisions.margin_plan.time_held` with `TIME_HELD` and
`CONSTANT`, and the `margin` arm's variant `margin held`; changes no default and no other
variant's plan.

## Context

`margin curved noise` (ADR 0091) plans on each channel's margin at the status quo and its curve,
both read off the weekly history by regression. On most of the bench's worlds that reading is
right often enough to gain. On one family it is not, and the regression cannot know it.

**The twins.** The bench's twin worlds (family 17) hold two channels. Television's weekly spend
rises from 30 to 100 over three quarters of the history and stays there, each week's spend
within a log-scale of 0.05 of that level; search's moves freely. In one twin television's
response is a fixed S-shaped curve of its spend. In the other it is a straight line whose slope
drifts over the weeks, set so that along the ramp both twins return the same. The two histories
differ only by how far each week's spend lies off its level: the conflation of a time-varying
effect with a curve that Dew, Padilla and Shchetkina (arXiv 2408.07678) measure. Their best
plans differ: television's best weekly spend is 94.69 in the curve's twin and 110.81 in the
line's, from a status quo of 100, and each twin's best plan costs 0.095 and 0.061 of a euro of
the budget in the other. One plan played in both twins regrets at least 0.01893 a euro on their
average; the status quo regrets 0.01924.

**What the noise variant does there.** A study written apart from this package read the noise
variant on the twins. On 40 tuning worlds its regret less the status quo's was +0.1307 a euro
[+0.074, +0.193], and on 100 fresh worlds +0.1077 [+0.071, +0.151]: +0.1695 in the curve's twin
and +0.0458 in the line's. On those 100 worlds it sent television below 60 on 11 and above 105 on
61. It fits one fixed curve to a history a drifting line explains as well.

**No reading of the posterior separates the twins.** By the posterior's own measures television
there is a typical channel. Its shrunk margin's coefficient of variation is 0.31, 0.36 and 0.44
at the 10th, 50th and 90th percentile, where the other families' medians run from 0.22 to 0.55.
The z-score of the planned gain is 0.08 to 0.70 there and 0.06 to 1.67 elsewhere. The error is a
misspecification, a fixed curve against a drifting line along a ramp, and the covariance does not
carry it. The study screened 45 planners on 1930 tuning worlds under a rule declared before any
regret was read. Every planner that reads the posterior either left the twins' loss or gave the
noise variant's gains back on the other families. Four rows show it; the give-back is the mean
over the 15 other families of the planner's regret less the noise variant's:

| planner on the noise variant's posterior | twins, less the status quo [95 %] | give-back [95 %] | other worlds whose plan changed |
|---|---|---|---|
| the noise variant itself | +0.1307 [+0.074, +0.193] | 0 | 0 of 1890 |
| mean-variance, weight 3 | +0.0573 [+0.036, +0.081] | +0.0065 [+0.0031, +0.0103] | 1559 |
| lower confidence bound, the mean less 0.5 deviations | -0.0007 [-0.007, +0.006] | +0.0356 [+0.0286, +0.0430] | 1737 |
| a channel of a large posterior coefficient of variation held, past 0.3 | +0.0037 [-0.001, +0.011] | +0.0812 [+0.0717, +0.0914] | 1622 |
| the time guard at 0.8 | 0, the status quo on 40 of 40 | 0 | 0 of 1890 |

The mildest posterior planner that brings the twins within 0.010 of the status quo is the lower
confidence bound at 0.5 deviations, and it gives back 0.036 a euro a family. Pessimism strong
enough to stop the move on the twins stops the good moves everywhere.

**The design separates them.** The absolute correlation of television's weekly spend with the
week's number is 0.957 to 0.977 on the twins. On the 3715 other worlds the study read, the
largest correlation of any channel is 0.468, on a 52-week history.

## The identification argument

Write a channel's response in a week as `h(x, t)`, its spend `x` and the week `t`. A history
reads `h` only at the points `(x_t, t)` it logged. Where the spend is a function of the week,
`x_t = g(t)`, those points lie on one path, and the history gives `h(g(t), t)` along it and
nothing off it. A curve fixed in time, `h(x, t) = f(x)`, and a line that drifts,
`h(x, t) = b(t) x` with `b(t) = f(g(t)) / g(t)`, agree on the whole path. A plan needs the
slope across the path, the return of a euro more at the status quo in the planned weeks: `f'(s)`
in the first and `b(T)` in the second. The history does not hold it. Any plan computed from the
history is the same in both worlds, so no estimator and no measure of its uncertainty can be
right in both; what tells them apart is a spend that moves off the path, which is an experiment.
Noise about the path helps only as far as it reaches: on the twins a log-scale of 0.05.

This is a property of the design, so the guard reads the design. A spend that follows the week
closely has a high correlation with the week's number, whatever the response.

## Decision

- **`mixdyn.decisions.margin_plan.time_held(spend)`** returns, for a history
  `(weeks, channels)`, each channel's absolute Pearson correlation of its weekly spend with the
  week's number over every week, and the channels where it lies above `TIME_HELD`, 0.8. It reads
  the spend alone: no margin, no fit, so no rounding sign can decide it.
- **A constant spend reads 0.** A channel whose spend ranges over at most `CONSTANT`, 1e-9, of
  its mean's size is a constant: its correlation is 0 and it is not named. Without that a level
  spend whose last bits drift with the week would read a correlation of 1.
- **The variant `margin held`** (`scorecard.arms.margin`) is `margin curved noise` with the
  guard, in this order.
  (1) The fit is `margin curved noise`'s. A channel it names spanned (ADR 0091) has a margin of
  exactly 0 and its box is its status quo, as in that variant.
  (2) Every channel's margin is shrunk as in that variant. A channel the guard names keeps its
  margin in the shrinkage: the common level the other channels are pulled towards is the one that
  was read, and the guard changes the box alone.
  (3) A channel the guard names has its floor and its ceiling set to its status quo. A channel
  that is spanned too is held as spanned, and the guard does not count it.
  (4) The other channels are water-filled in their box at the budget. Where the guard holds a
  channel and fewer than two channels are free, or the box holds no plan at the budget, the plan
  is the status quo and the claimed gain 0. A channel is free where it is not held and its box
  is not a point; the count reads no margin.
  (5) Where the guard names no channel the plan is `margin curved noise`'s, by the same calls.
- **The record** adds each channel's correlation (`time correlation`) and the count of free
  channels to its flags, and the guard's threshold to its settings. Its settings name each held
  channel with its reason: the correlation for the guard's, the span for the fit's.
- **`margin curved noise` stays**, as the rule that was read before the guard and the parity
  target of an audit. `margin held` runs the regression again in its own plan, so its cost and
  its failure are its own.
- **The name** is one word after `margin`, as `margin guarded` and `margin ranged`: the variant
  is the arm's planner with every hold it has.

## Consequences

- *The measured effect, by the study that froze the guard.* On the twins the plan is the status
  quo on every world read: 40 of 40 tuning worlds and 100 of 100 fresh ones, where the noise
  variant regrets 0.13 and 0.11 a euro more than the status quo. Off the twins the guard named no
  channel on any world: the plan is the noise variant's on 1890 of 1890 tuning worlds, on 1365 of
  1365 check-split worlds, on 60 of 60 worlds of the pilot's box and on 400 of 400 fresh worlds
  of three other new families. So every cell the noise variant was read at is this variant's,
  the twins' apart.
- *Parity.* One function plans both variants, and the guard adds only the hold and the two
  conditions on it. So where the guard names no channel the variant makes `margin curved noise`'s
  calls on the same arguments, and its plan, its claim and every flag and setting the two share
  are equal to the bit by construction; the unit tests check it on three fixture worlds.
  The variant's records were made by the arm's harness beside the plan of the code the guard was
  frozen in, run in that code's own environment on the same exports. On the twins' 140 pilot
  worlds, 70 seeds read in both twins, both play the status quo on 140 of 140 and name
  television alone, its correlation 0.957 to 0.977 and equal in the two to the bit; the claimed
  gain is 0 on all, and `margin curved noise` plays the status quo on none. On 160 pilot worlds
  of three other new families neither names a channel, the largest correlation being 0.27, and
  the variant's plan and claim are `margin curved noise`'s to the bit on 160 of 160. Against the
  frozen code the plan is equal on the 100 worlds of two of those families. On the 60 worlds of
  the family with a channel whose spend never moves it differs on 8, by up to 5.1 % of the
  weekly budget: that code leaves the spanned channel's margin to rounding, the difference
  ADR 0091 measures.
  The same read over the 3155 worlds `margin curved noise` was read on, 1730 tuning worlds,
  1365 check-split worlds and the 60 of the pilot's box, compared the plans alone and read no
  regret. Neither code names a channel on any world: the two codes' correlations are equal to
  the bit, and the largest of any channel is 0.468. The variant's plan and claim are
  `margin curved noise`'s to the bit on 3155 of 3155. Against the frozen code the plan differs
  by at most 2.6e-13 of the weekly budget, and no record failed.
- *What it cannot gain.* On the twins the variant equals the status quo, so its regret less the
  status quo's is 0 with an interval of [0, 0]. No rule gains there on the history alone.
- *The cost.* One correlation a channel beside the noise variant's regression, which the variant
  runs again: its plan costs what `margin curved noise`'s does (ADR 0091), an order of magnitude
  of 0.3 to 0.4 CPU-s a world on a loaded laptop.
- *Limits.*
  (1) The threshold is not calibrated. On the worlds read nothing lies between 0.47 and 0.957, so
  every threshold between them plays the same plans. Without noise
  (`validation/time_guard.mac`), a spend that steps once at the share `a` of the history has a
  squared correlation of `3 a (1 - a)` and is held where the step lies between 31 % and 69 % of
  the history; a spend that rises in a straight line over the first share `b` and is level after
  it has `b (3 - 2 b)^2 / (4 - 3 b)` and is held from 36 %. Whether such channels should be held
  is a decision about the product, and worlds with ramps of several lengths would place the
  constant.
  (2) Three or more channels are tested by a unit test only. The twins have two, so every world
  the guard fired on was the status quo; no read has scored the other channels' plan beside a
  held one, nor the choice to keep the held margin in the shrinkage.
  (3) The guard reads a monotone path. A spend that follows the season or the price is a function
  of time too, and its correlation with the week's number is low.
  (4) The tested rung's arm is not guarded by this decision.
- *The tests* (`tests/test_margin_plan.py`, `tests/scorecard/test_margin_arm.py`). The
  correlation is the definition's, its sign dropped; a spend at 0.8001 is named and one at 0.7999
  is not, rising or falling; a level spend whose last bits drift reads 0; a step and a ramp are
  named where their closed forms say, the step's exactly over the weeks. In the arm, on a pair of
  twins the plan is the status quo and the settings name television and its correlation; on
  worlds where no spend follows the week the plan, the claim and every shared flag and setting
  are the noise variant's; with one of three channels held the other two are planned at the
  budget in their boxes, every margin shrunk as the noise variant shrinks it; with one channel
  left free, or a free box without a plan, the plan is the status quo; a box that is a point is
  not free; and a flat channel is held as spanned, not by the guard. Each of these tests fails on
  a mutant of what it names. `validation/time_guard.mac` proves the two closed forms and their
  reach under Maxima 5.46 and 5.50.

## Alternatives considered

- **A hold by the regression's own columns**: a channel whose adstock the nuisance columns
  reproduce with an R-squared above 0.95. It removes the twins' loss too, and it would also catch
  a channel that follows the season or the price. It changed 39 other tuning worlds, all 52-week
  histories of one family, at +0.0037 on that family's mean and inside the screen's slack. The
  declared tie-break took the guard that changes no other plan. It has had no final read.
- **A box graded by the correlation**, each channel's half-widths times `1 - corr^2`, in place
  of a hold. It was read on the tested rung's rule, where television's history and its
  correlation are the same: on the twins' 40 tuning worlds it left +0.0010 a euro
  [-0.002, +0.005] against the status quo, where the hold leaves 0. It acts on every channel of
  every world, so it would move plans the guard leaves alone, by a function nothing calibrates.
- **Pessimism on the posterior**: a lower confidence bound, a mean-variance objective, a trust
  region scaled by each channel's deviation, a blend with the status quo, a hold by the posterior
  coefficient of variation. The table above: each left the loss or gave the gains back.
- **A lower threshold.** At 0.5 the plans are the same on every world read. 0.8 is the first of
  the two the study declared, and the further from the largest correlation seen off the twins.
- **Guarding `margin curved noise` in place.** That variant is the rule as it was read on the
  check split and what an audit compares the research code with; a new variant leaves it as it
  is.
