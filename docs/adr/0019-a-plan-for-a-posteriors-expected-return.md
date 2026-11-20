# 0019. A plan for a posterior's expected return, over its draws

Status: accepted, 2026-10-04; amended 2026-10-05 by ADR 0034: the plan is the best split of the
means' interpolants on the grid, searched by branch and bound, not the envelopes' where a mean is
S-shaped; amended 2026-11-06 by ADR 0071: a target marginal return or a target return on spend may
take the budget's place.

## Context

ADR 0010's planner reads each channel at its posterior medians, and left a plan for the posterior's
expected return over its draws to be measured against it. ADR 0011 measured the two on Track M v2;
the bench kept the expected return by the rule it had stated, and its families' run planned the Hill
and the `tanh` fits' expected returns stacked by their weights. That planner was never the engine's.
It lived in the harness that scored the engine on the bench (`evidence/bench/track_m2_bayes.py`,
which plans by `plan_posterior` since this decision), and `plan_geos` (ADR 0005) plans one
`chc.response.Channel` a cell, the medians' that `BayesFit.channels()` hands it.

The value of a lift test is a difference of expected returns: the posterior is re-planned after
its draws are reweighted by what a test might read, many times for each candidate. The planner it
needs takes draws and their weights as arrays, and is the bench's, so that what the bench measured
is what the engine plans.

A posterior's expected worth is separable in its channels, but a channel's mean worth over the
draws is no `chc.response.Channel`: it mixes kernels and curves. `chc.allocation.allocate` plans
one channel set exactly, and CHC's `cvar_allocate`, which plans a set of readings, is not in the
CHC release the engine pins (0.10.0).

## Decision

- **`Readings`**: a posterior's draws as a planner reads them, each channel's retention,
  half-saturation point, Hill slope (NaN where the curve is a `tanh`) and coefficient,
  `(draws, channels)` in the panel's units; each draw's curve, fit and chain; each fit's weight,
  `stacking`; the fits' leave-one-out log predictive densities where the weights were solved from
  them, so they can be solved again; and the kernels' length. A draw's weight, `weights`, is not
  stored: it is its fit's weight over the fit's number of draws, so the draws' weights cannot
  disagree with the fits'. What no posterior holds is refused, and the arrays are read-only.
- **`readings(fit, *, draws=None)` and `stacked(fits, *, weights=None, draws=None)`** read one fit
  or several fits of one geo of one panel (the same channels, calendar, KPI, fitted periods,
  observations and kernels' length), each fit's `draws` evenly spaced over its chains as
  `BayesFit.channel_draws` spaces them, or every draw. The fits' weights are given, or stacking's
  (ADR 0011) on their leave-one-out densities: one fit's is 1; two fits' are solved by bisection
  on the objective's slope, which always ends, and fits equal at every observation share the
  weight evenly; three or more by Cover's iteration, which can fail to converge, and then
  `stacked` raises.
- **`worths(panel, geo, readings, periods, limits, *, grid=512)`**: each draw's worth of each
  channel at `grid` spends a period across its range, what the channel returns over the plan's
  periods and the kernel's length after them, the geo's spend in the panel carried in: the worth
  `chc.allocation` reads.
- **`plan_posterior(panel, geo, readings, budget, periods, limits, *, grid=2048)`** reads each
  channel's mean worth over the draws, each fit's mean weighted by its weight, at `grid` spends
  across its range, and spends the budget from each channel's least down the upper concave
  envelopes of the means, the steepest stretch first. Where a mean is concave in its spend the
  plan's expected worth is within `sum_c h_c^2 max|m_c''| / 8` of the best plan's, `h_c` the
  grid's step; where it is S-shaped the plan is the envelope's. `PosteriorPlan` is `GeoPlan`'s
  shape for one geo: the spend a period, the current one (the mean over the panel's last `periods`
  observed periods), the budget, and the plan's and the current spend's expected gains against
  spending nothing in the plan, so the history's carryover counts in neither.
- **The arithmetic is NumPy's, in double precision whatever JAX's**, and it is the bench's
  planner's operation for operation, so the bench's plans are made again bit for bit.
- Every range needs a finite most, as in ADR 0005.

## Consequences

- *The oracle* (`tests/fixtures/oracles/posterior_plan_bench.json`) fitted three of Track M v2's
  pilot worlds, the drawn environment's 904 and 910 and the reference's 907, each by a
  Hill and by a `tanh` without the geo tests, kept 400 draws of each fit evenly over its chains,
  and recorded them with the plans a verbatim copy of the bench's planner made of each fit's
  expected return and of the two stacked by Cover's weights: on the machine that recorded them the
  library makes the nine plans bit for bit, and the tests hold them to `1e-12`; its means at every
  64th spend of each grid are the planner's to a relative `1e-12`. The worlds' stacking weights
  are 0.84, 0.33 and all but 0 on the Hill; on the drawn worlds every plan holds two channels at
  their least, and on the reference world it splits the budget between two channels.
- *What the tests hold besides*: each draw's worth is `chc.response`'s, run over the history, the
  plan and the kernel's tail, to a relative `1e-12`, also under kernels longer than the history;
  the plan of one `tanh` draw falls short of `chc.allocation.allocate`'s exact plan by no more
  than the grid's bound, under a millionth of its gain; an S-shaped mean is planned on the least
  concave majorant a brute force finds; a plan's gains are the draws' weighted mean gains read
  through `chc.response`; and two fits' weights by bisection meet Cover's certificate, reach
  Cover's objective, are an end whose slope does not point inward or else the last float below
  the slope's root, and do not move when each observation's densities move by one constant, down
  to where `exp` underflows.
- *Mutations*, each run on a copy of the tree: of 41 of `decisions.posterior`, the tests kill 40;
  the 41st keeps collinear points on an envelope, splitting a stretch into two of its slope to the
  last bits. Of 9 of the two fits' bisection, they kill 8; the 9th takes the lower end only where
  its slope points outward, and where the slope there is 0 the bisection, which keeps the float
  below the root, returns that end exactly.
- *The bench's harnesses* call `plan_posterior` in place of their own copy of it. On the oracle's
  three worlds, each fit replaced by its recorded draws, the three harnesses' records are the same
  value for value before and after, 18 of their 30 plans the posterior planner's. The library
  skips a stretch of no length and stops before a take where nothing is left, and the copy did
  neither; that matters only where a channel's range is a point or the budget is the ranges'
  least, and none of the 360 worlds Track M v2 scores, pilot or not, has either. The no-test risk
  harness keeps its own readings for `cvar_allocate`, which takes `chc.response` channels and
  whose draws it shares between the two fits by the Hill's weight times 400, rounded.
- Cover's iteration exhausted its million iterations on one of the families' 300 scored worlds
  (drawn 10091), whose stacked arm was then scored at the status quo. It exhausts them too on three
  observations built so that the objective's slope is exactly 0 at an end, which bisection takes.
  Bisection on two fits always ends: at an end, or at the last float below the slope's root.
- A plan is of one geo: the posterior of a model over geos, or a plan with totals over geos or
  channels, is not read here. No bound on the envelope's shortfall is reported; on S-shaped means
  the grid sets how near the envelope's plan is to the best.

## Alternatives considered

- **The medians' plan**, `BayesFit.channels()` into `plan_geos`: the plug-in. Kept as it is, but
  it is not the decision for the posterior's expected return, which the value of a lift test
  differences.
- **CHC's `cvar_allocate` at level 1**, which plans the mean of a set of readings and is exact on
  concave curves. Not used: it weighs every reading alike, so stacking's weights would be draws
  repeated; it reads each reading on its own envelope rather than the mean's; it plans by cutting
  planes over every reading, for every candidate test; and it is not in the CHC release the engine
  pins. ADR 0029 plans a posterior's worst share by it, the draws shared out by the fits' weights.
- **`weights` held as a field beside `stacking`**: rejected for a derived one, so that no
  `Readings` can hold draws' weights that are not its fits' spread evenly.
- **Cover's iteration for two fits**, as for more. Rejected: two fits' problem is one-dimensional
  and concave, and bisection on its slope ends where Cover's may not.
- **JAX's arithmetic**: rejected. The planner then differs from the bench's in the last bits, and
  in single precision, JAX's default, by far more.
