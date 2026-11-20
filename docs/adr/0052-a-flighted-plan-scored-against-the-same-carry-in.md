# 0052. A flighted plan scored against the same carry-in

Status: accepted, 2026-10-28; adds to ADR 0019's plan for a posterior a schedule over the periods,
read with ADR 0016's rule that the planner reads the channels at the posterior medians; amended
2026-10-28: a draw of weight 0 enters neither the median nor an interval, and an interval's ends
are ADR 0050's weighted quantiles

## Context

Every plan of the engine is one spend a period a channel. A planner also asks when to spend: in
pulses or continuously, and how to pace a budget over weeks. `chc.allocation.allocate(...,
schedule=True)` searches a spend for each period of each channel within its box, the periods
together spending the budget. Where every kernel is one period long it is the best schedule to
the search's tolerance; where a kernel is longer it is the best of local ascents, its bound the
curves' concave envelopes'. The search plans one set of channels, not a posterior.

Two ways of scoring a schedule get it wrong, and a planner that reads them recommends pulses that
lose:

- the carryover the history leaves in the plan's periods credited to the plan and not to the plan
  it is compared with, so that a plan that changes nothing reads a gain;
- a pulse compared with a status quo or a plan read on other terms than the continuous plan at
  the same budget, so that on a concave curve, where spreading the spend evenly returns most, a
  pulse reads as a win. On `10 sqrt(x)` with no carryover, 440 over eight periods returns
  `8 x 10 sqrt(55) = 593.3` at 55 a period, and `4 x 10 sqrt(10) + 4 x 10 sqrt(100) = 526.5`
  pulsed at 10 and 100.

## Decision

- **`mixdyn.decisions.flighting.plan_flighting(panel, geo, readings, budget, periods, limits, *,
  reading="median", level=0.9)`** plans one set of channels, read off the posterior at each
  parameter's weighted median over the draws of weight above 0 (each draw weighted as
  `Readings.weights` weighs it, a tie between two values averaged as `np.median` averages an even
  count's middle two), or at one draw's index. A draw of weight 0, a stacked fit of weight 0's, is
  dropped before the median: averaged in at a tie, it moved the plan. Fit 0 weighted 1 with
  half-saturation points 0.5 and 4, fit 1 weighted 0 at 1: the median read 0.75 where fit 0's is
  2.25, and the schedule of 4 over four periods within 0 to 2 a rate of 1 where it pulses 2, 2,
  0, 0. The draws of weight above 0 must all be of one curve. It calls `allocate(..., schedule=True)` and `allocate(...)` on them: the schedule
  and the best rate a period, the continuous plan.
- **The schedule is never worse than the continuous plan at the reading.** Where every kernel is
  one period long, chc's search of the schedule closes its gap on interpolants of the curves, and
  its schedule can return less than the best rate a period on the curves themselves: on a Hill of
  slope 0.30, steep at 0, 30 over three periods was scheduled 10.0024 in the first period, 3.5
  times the search's tolerance short of 10 in each. Where the schedule returns less than the
  continuous plan at the reading, the plan holds the continuous plan's rate in every period.
- **Three plans scored at every draw, each against spending nothing in the plan.** The schedule,
  the continuous plan and the status quo, each channel's mean spend over the geo's last `periods`
  observed periods held in every period, are scored at each draw's own retention, curve and
  coefficient: what each adds to the KPI over the plan's periods and the kernel's length after
  them, the history carried into all three and into the plan of nothing alike. The carry-in
  cancels in every comparison, so a plan that keeps the status quo gains exactly what the status
  quo does, at every draw. The result holds the three gains a draw, their weighted means and
  central `level` intervals, and the gain of the schedule over the continuous plan at each draw.
  An interval's ends are ADR 0050's weighted quantiles, each a draw's own value: only the median
  averages at a tie.
- **Flighted only where every reading in the interval says it pays.** The schedule is recommended,
  `"flighted"`, only where the lower end of the weighted central `level` interval of its gain over
  the continuous plan is above the schedule's search tolerance; else the plan is `"continuous"`.
  On concave curves through one-period kernels the best schedule is the best rate a period, so the
  recommendation there is always continuous.
- **The local search is said, not hidden.** Where a kernel is longer than one period, `allocate`
  logs a warning with `chc_event="allocation_local"` and reads `limit="local"`, `stopped="cap"`.
  The plan passes the log record on as chc logs it, and holds `limit`, `stopped`, `worth`, `bound`
  and `tolerance` of the schedule's search. For a user, `limit="local"` means: the schedule is
  never worse than the best rate a period at the reading, and is the best of several ascents, but
  it is not proved the best; `bound - worth` is the envelopes' slack, which chc reports as loose
  by much more than the schedule falls short. The recommendation does not read the bound.
- **Pacing is refused.** A budget given a period, an array, is refused with a `ValueError`: chc's
  planner spends one budget over the periods, each period's spend free within its box. Pacing
  would need a constraint a period, or a group of periods, on the sum over the channels, in the
  schedule's search: in its boxes' fill where every kernel is one period long, one budget a period
  with its own shadow price, and in the ascents and the Frank-Wolfe bound where a kernel is longer.
  Neither is in chc; a period's box, `lower` and `upper` a channel, is the only per-period limit
  the planner reads.

## Consequences

- *Properties* (`tests/test_flighting.py`). One channel, a Hill of half-saturation 100 and slope
  1, retentions about 0.5, four periods each at the current 100: the schedule's and the continuous
  plan's gains over the status quo are exactly 0 at every draw. On `c sqrt(x)` read by a Hill of
  slope 1/2 whose half-saturation, `10^12`, lies far past the spend, 440 over eight periods within
  10 to 100: the plan is continuous, the schedule returns at least the continuous plan's 593.3 at
  the median to `1e-9`, and every draw's schedule returns more than the 10-and-100 pulse. The
  schedule spends the whole budget wherever the box allows it. Each draw's score is
  `chc.response`'s at that draw's retention, and the schedule is `allocate`'s at the median's.
  On an S-shaped Hill, slope 3, where 1.2 over four periods returns 0.105 at one rate and 0.633
  in one period, the pulse is flighted. On random concave channels through one-period kernels the
  schedule's worth at the median is never below the continuous plan's, which chc documents and
  the plan enforces, and the plan is continuous.
- *Mutations*: a hand pass of the module's mutants, each failing a test: the carry-in dropped from
  the status quo's score or from the plan of nothing's, the recommendation read off the interval's
  upper end or against the status quo, a share of the budget planned, a retention fixed in the
  scores or in the plan, chc's record stopped from reaching the caller, the median read as a mean
  or unweighted, the refusals dropped, the kernel's tail dropped from the scores, and the
  schedule kept where it returns less than the continuous plan.
- The schedule is planned once, at one reading, and scored at the rest: a schedule that pays at
  the median and at no other reading is not recommended, but no schedule better at every reading
  is searched for.
- `chc.allocation`'s schedule is experimental in chc; its numbers may move with chc's releases.

## Alternatives considered

- **Each draw's own schedule.** Rejected: one search a draw costs the draws' count of searches,
  each local where a kernel is longer than one period, and leaves no single plan to run, only a
  distribution of schedules whose mean is no draw's best.
- **One schedule searched for all the readings at once**, the most expected gain over the draws,
  as `cvar_allocate` plans one rate a period for the worst share of them. Not in chc: its schedule
  search reads one set of channels. It would be the plan to adopt where chc grows one.
- **The plan at the posterior means.** Rejected for ADR 0016's reason: the medians are what every
  other plan of the engine reads, and a mean of a heavy-tailed half-saturation or coefficient
  plans at a channel few draws hold.
