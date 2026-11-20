# 0034. The posterior's plan searched on its grid

Status: accepted, 2026-10-05. Amends ADR 0019 (an S-shaped mean is searched, not planned on its
envelope) and ADR 0030 (a test's value is read on the plans the search makes). Amended 2026-11-06
by ADR 0071: the search plans a total within a stretch less a charge a unit, a fixed budget the
stretch of one point and no charge, bit for bit as before.

## Context

ADR 0019's `plan_posterior` reads each channel's mean worth over the draws at `grid` spends and
spends the budget down the upper concave envelopes of the means, the steepest stretch first. Where
the means are concave that fill is the best split of their interpolants. Where one is S-shaped the
fill can stop inside a stretch of its envelope that passes over points of its grid, a chord that
stands above the mean, and the plan, read on the mean, is the envelope's. ADR 0019 said so, and
that on S-shaped means the grid sets how near the plan is to the best; it does not, the chord does,
however fine the grid. ADR 0030 plans each outer draw's posterior, and perfect information's each
draw, by the same fill, and one Hill draw of a slope above 1 is S-shaped wherever its range passes
the curve's inflection.

Measured on the scorecard's pilot of the drift family (`evidence/plan_exactness.py`, its record
`evidence/results/plan_exactness.jsonl`: the stacked arm's 120 records, each record's readings
rebuilt from the draws it kept, its 2048-point grid, and the best split of the interpolants by brute
force): 60 of the 120 records hold a mean that stands below its envelope somewhere, 78 of the 360
channels' means. On 9 the fill is not the best split: its worth on the interpolants falls short of
the best by 0.0004% to 3.2% of it, above 0.5% on four, and its expected gain, read on the means off
the grid, by up to 3.5%. On the other 111 it is the best split.

## Decision

- **The plan is the best split of the means' interpolants on the grid**, to a share `1e-9` of its
  bound, found by branch and bound over boxes of each channel's grid points (Udell and Boyd 2016,
  on a grid). A box's bound is the fill down the upper concave envelopes of the box's points, each
  channel from its box's first point, and that fill, read on the interpolants, is a plan. A box
  whose fill stops inside a stretch that passes over a point is cut in that channel, at the first
  point at or past the fill's spend, into the two boxes that share the point; a box the budget
  cannot meet within its points is dropped. The box of the largest bound is cut first. The search
  ends when no box's bound is above the best plan's worth by more than `1e-9` of the first box's,
  or at 500 boxes, where it logs a warning with the plan's worth and the bound left and returns the
  best plan it found.
- **Where the first box's fill stops on its envelopes' points, that fill is the plan**, and nothing
  is cut: on concave means the plan is ADR 0019's bit for bit.
- **The search is finite and its answer exact on the interpolants.** In each cell of the grids the
  worth is linear, so a best split has every channel but one on a point of its grid; a box is cut
  only across a stretch that passes over a point, so every cut leaves fewer points in a box.
- **Each channel's envelope over a box's points is computed once** and kept for the boxes that
  share it, so a cut costs one channel's envelope and one fill.
- **The value of a test** (ADR 0030) plans each outer draw's updated posterior and each draw's
  perfect information by the same search, so it is the value of a test to the plan the engine
  makes.

## Consequences

- *The tests*:
  - ADR 0019's oracle: the nine plans of the bench's planner are still made to `1e-12`, as their
    fills stop on their envelopes' points;
  - on 300 cases of two and three channels whose means are Hill curves of slope 0.5 to 5 tilted by
    a line, on grids of 41 and 13 points, and a budget anywhere the ranges can spend, the plan's
    worth on the interpolants is a brute force's to `2e-9` of it, and the plan spends the budget
    within the ranges;
  - two Hill means of slope 3 at half-saturation points 1 and 1.01, each range 0 to 1.6 a period
    and 1.6 a period to spend, CHC's example of its own envelopes' shortfall: the fill splits about
    1.27 / 0.33 for about 0.705 a period, and the plan spends all on the first,
    `1.6^3 / (1.6^3 + 1)` = 0.8038 a period, to `1e-12`; stopped at one box it returns the fill and
    warns;
  - an S-shaped television beside a nearly linear search, whose steepest stretch is steeper than
    search's, is given nothing, the brute force's best.
- *Mutations*, each applied alone to the planner and run against the planner's tests with a fresh
  Hypothesis database: 15, of the search (5: no search, the heap's order, a better plan not kept,
  a plan's worth read on its bound, a gap of a tenth), of a box (4: its fill and its bound from the
  grid's first point, every box's envelope the whole grid's, envelopes kept by their first point
  alone), of the cut (3: the cell below it lost, either child dropped), boxes the budget cannot
  meet filled, the bound without the fill's slopes, and the warning at the cap. All 15 failed a
  test. A first pass left the bound from the grid's first point alive, as the search finds its
  plans before it prunes on that bound; a property test of a box's bound, that it is at least
  every split's in the box by brute force, was added for it.
- *The pilot's 120 records*: the search's plan is the brute force's on every one, to `2e-14` of
  its worth, and it raises the expected gain on the nine of the context, never lowering it on any.
  It filled one box on 79 records, 3 to 5 on 37, and 7, 10, 13 and 35 on the last four, far from
  the cap; 32 of the 41 records it cut on kept the first fill, the cuts proving it the best. Not
  timed, as the machine was loaded: a plan costs one fill a box, and one box where the means are
  concave.
- Sums of S-shaped curves under a budget are NP-hard to plan, so nothing short of the cap bounds the
  count of boxes; a plan that reaches it is the best found, and the warning says by how much it may
  fall short. `PosteriorPlan` holds no bound, as before.
- *The bench's harnesses* call `plan_posterior`, so their plans of S-shaped means change with it.
  Track M v2's recorded results were made by the fill and stay as recorded.

## Alternatives considered

- **CHC's `allocate`**, which searches S-curves by branch and bound since its ADR 0051: a channel's
  mean worth over the draws is no `chc.response.Channel` (ADR 0019), and that search is in no CHC
  release the engine pins. Rejected.
- **A finer grid**: the chord stands above the mean however fine the grid. Rejected.
- **A max-plus programme over the grids' points**, as the bench's truth searches a non-concave
  quarter: exact on the points only, and `O(C G^2)` for every plan, concave or not, where the search
  costs one fill on concave means. Rejected.
- **A bound on `PosteriorPlan`**: the search closes to `1e-9` of its bound but at its cap, which it
  logs; a field every caller would read to ignore. Not added.
