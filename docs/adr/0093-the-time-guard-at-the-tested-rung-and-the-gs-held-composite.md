# 0093. The time guard at the tested rung: the variant `anchored held` and the composite `gs held`

Status: proposed, 2026-11-17; adds the `anchored` arm's variant `anchored held` and the composite
`gs held`; changes no default and no other variant's plan.

## Context

The bench's family 17 holds twin worlds (`causaldyn_bench.scorecard.twins`). Each has two channels,
television and search. Television's weekly spend rises from 30 to 100 over the first three
quarters of the history and holds there. In one twin television's response is a fixed S-shaped
curve; in the other it is a line whose slope drifts with the week, so that along the ramp both
twins return the same. One seed draws the same history in both. Their best plans differ: from a
status quo of 100 a week, television's best spend is 94.69 in the curve's twin and 110.81 in the
line's.

The curved tested-rung rule (ADR 0089) loses there. Its research implementation, written apart
from this package, was read on 40 worlds of the family, the pilot block's worlds 30 to 49 of each
twin, at 4 tests a channel. Mean regret a euro less the status quo's, with a 95 % bootstrap
interval:

| family 17, 4 tests a channel | the curve's twin (20) | the line's twin (20) | both (40) |
|---|---|---|---|
| the rule's own plan | +0.1927 [+0.117, +0.288] | +0.0802 [+0.005, +0.179] | +0.1365 [+0.079, +0.201] |
| television's floor and ceiling at its status quo | 0 (the status quo on 20 of 20) | 0 (20 of 20) | 0 |
| the box's half-widths about the status quo times `1 - corr^2` | +0.0060 [+0.001, +0.010] | -0.0041 [-0.010, +0.002] | +0.0010 [-0.002, +0.005] |

The status quo's own mean regret there is 0.0103 and 0.0282 a euro. The rule's readings route 7 of
the 40 worlds to `stacked`; the table reads the rule's own plan on all 40.

The tests do not help, and the reason is one of identification. A go-dark test takes a channel's
spend off and reads the KPI that leaves: the channel's whole contribution at the spend of its
weeks. Along the ramp the twins share that contribution, so their tests read the same. The plan
needs more than the contribution: it needs the margin at the status quo and how that margin
falls, the channel's curve. The rule reads the curve off the tests at their several spends and off
the regression's margin. Where a channel's spend is a function of the time, a test at a higher
spend is a test at a later week, and a response that curves along the spend and one that drifts
along the time give the same tests and the same history
(`mixdyn.decisions.margin_plan`, the time guard). No reading the rule has places such a channel
on its curve.

The margin rule meets the same twins without tests. The guard built for them reads the spend
alone: `mixdyn.decisions.margin_plan.time_held` names a channel whose weekly spend correlates
with the week's number above 0.8 in absolute value, for a planner to hold at its status quo. The
bench's tests run apart from the history, so a channel's correlation is the same at every rung.

## Decision

- **The variant `anchored held`** (`scorecard.arms.anchored`) is `anchored curved` under that
  guard. It reads `time_held` on the export's whole history of weekly spend. The constant, 0.8, is
  the guard's own; it was not tuned here. Four points define the variant.
  1. *A held channel stays at its status quo.* Its floor and its ceiling are its status quo in the
     water-filling, and in the corner where no curve is read. The other channels are planned at
     the budget in their box, along the same curves.
  2. *Too little is left to plan: the status quo.* Where fewer than two channels are free, a free
     channel being neither held nor boxed to a point, or where the box so cut holds no plan at the
     budget, the plan is the status quo. It claims a gain of 0 with the interval (0, 0).
  3. *A world with a held channel is never routed.* `stacked` plans every channel and cannot hold
     one. Where the fit's readings route such a world, by the slope or by the kernel, the plan is
     the status quo, and the record's `routed` is false, so that a composite plays the variant
     there. The record keeps the two reasons as the fit read them and says the route was
     withheld.
  4. *A world without a held channel is `anchored curved`'s.* The same code gives the plan, the
     claimed gain, its interval and the route. The guard adds flags and settings only.
- **The record.** Its flags add each channel's correlation with the week's number
  (`<channel> time correlation`), whether any channel is held (`time held`) and whether a route
  was withheld (`route withheld`). Its settings add the threshold (`time_held`) and, under
  `held at the status quo`, each held channel with its correlation. The claimed gain is the gain
  of the plan returned. The guard reads the export's spend, its box and its status quo, and of
  the fit the route alone. It reads no margin, so no rounding of a margin can decide it. A spend
  that never moves has a correlation of 0 by `time_held`'s own tolerance.
- **The composite `gs held`** (`scorecard.composite.GS_HELD`) is `gs curved` with `anchored held`
  as its router. It plays `stacked` where that variant's record says the world is routed, and the
  variant elsewhere. It is assembled wherever `gs curved` is, with no reader changed:
  `scorecard.claim.read_run`, `scorecard.claim.joined_runs` and `scorecard.development.joined`
  assemble every composite of `scorecard.composite.COMPOSITES`. No record may take its name.
- **`anchored curved` and `gs curved` stay as recorded** (ADR 0089). They are the rule as it was
  read, and the parity target of this variant.
- **Defaults are unchanged.** A root that declares the `anchored` arm now holds three variants'
  records.

## Consequences

- *The development worlds.* The arm's harness made the three variants' records from the export of
  each world the curved rule's parity was read on (ADR 0089), the archived exports: M2's check
  split at the tested rung (1036 worlds), the pilot's 30 + 30 worlds of families 12 and 13 at
  k = 4, and M2's fit split at the tested rung (1339). It made them too from 160 worlds of
  families 18 to 20 generated at k = 4, the pilot block's worlds 30 to 49 of each of their 8
  environments. No plan was scored. No world of the 2595 has a held channel: the largest
  correlation of any channel with the week's number is 0.468 (family 7, a history of 52 weeks).
  On every one of them `anchored held`'s plan, its claimed gain, its interval and its route equal
  `anchored curved`'s to the last bit, and so does every other flag and setting the two records
  share. 663 of the worlds are routed, in both variants alike. No record failed.
- *The twins.* The 40 worlds of the Context's table were generated again at 4 tests a channel.
  `anchored held` plays the status quo on 40 of 40, claims a gain of 0 with the interval (0, 0)
  on each, and is routed on none. Television's correlation with the week's number lies between
  0.957 and 0.974 there, search's between 0.004 and 0.212. The fit's readings route 7 of the 40,
  all by the slope, and the variant withholds each of those routes. `anchored curved` moves the
  plan on all 40. No regret was read again: the plan is the status quo, whose regret the Context
  states.
- *The constant is not calibrated.* On the worlds read here no channel lies between 0.468 and
  0.957, so 0.8 parts the twins from every other world and nothing places it. Without noise the
  guard names a spend that steps once between 31 % and 69 % of the history, and one that rises in
  a straight line over 36 % of the history or more and then stays level
  (`validation/time_guard.mac`). A channel launched or scaled up in mid-history is held, and no
  world here says whether it should be.
- *What no world tests.* The twins have two channels, so every guarded world of the bench plays
  the status quo: a history of three channels or more with one held is covered by the unit tests
  alone. There the held channel's margin stays in the shrinkage, and the other channels' curves
  are read as in `anchored curved`. A routed world with a held channel plays the status quo
  because the engine's plan cannot hold a channel; whether that plan with the channel held would
  do better was not read. That a go-dark test cannot place a channel whose spend follows the week
  on its curve is shown on family 17 alone. And a box that holds no plan once a channel is held
  exists only where the export's status quo lies outside its box, which no export of the bench
  does; the variant plays that status quo all the same.
- *The cost.* The guard is one correlation a channel. The variant reads the curves in its own
  plan, as `anchored curved` does, so a root that runs the arm reads them twice a world; ADR 0089
  states what a read costs. No time was measured for this variant.
- *The tests* (`tests/scorecard/test_anchored_arm.py`), on the arm's eight fixture worlds, their
  histories edited to give a channel a chosen correlation with the week's number. Without a trend
  the held record is the curved one's to the last bit on all eight, and scored by the bench
  `gs held` plays what `gs curved` plays. Of two channels, the one that follows the week is named
  with its correlation and the plan is the status quo. Where the fit routes, by a rising return
  or by a kernel of 13 weeks, a world with a held channel is not routed, says its route was
  withheld and plays the status quo, and `gs held` plays the variant where `gs curved` plays
  `stacked`. With one channel of three held, a channel the curved plan raises and one it cuts,
  the held channel stays to the last bit and the plan is the water-filling's in the cut box at
  the budget; with one of four held the other three are planned. With one channel free, two held
  or one held and one boxed to a point, the plan is the status quo and the planner is not asked.
  A box whose ceilings fall short of the budget once television is held, and one with a floor
  above its ceiling among two free channels, play the status quo. A spend that never moves, or
  moves in its last bits alone, is not held. A channel at a correlation of 0.7999 is planned and
  one at 0.8001 is held. At k = 0 the held channel stays and the others take the corner. The
  claim's and the selection's readers assemble `gs held`, and no record may take its name. Each of
  27 mutants of the variant and the composite turns its named tests red.

## Alternatives considered

- **The engine's plan with the channel held, on a routed world.** It needs a planner in the
  engine's static arm that takes a channel's box, and a read of that plan on worlds that are both
  routed and held; the bench has too few. Later. Until then such a world plays the status quo.
- **A box graded by the correlation**, each channel's half-widths about the status quo times
  `1 - corr^2`. On the 40 worlds it read +0.0010 [-0.002, +0.005] a euro against the hold's 0
  (Context). It has no threshold, so it cuts every world's box a little: at the largest
  correlation read off the twins, 0.468, by 22 %. That gives up point 4, and with it the parity
  with `anchored curved` on every world the rule was read on.
- **Abstaining on the whole world**: the status quo wherever any channel is held. On the twins it
  is the same plan, since they have two channels. With three channels or more it gives up the
  plan of channels the guard has no reason to hold.
- **No guard at the tested rung**, family 17 a disclosed loss of `gs curved`. That remains the
  procedure if a later read of `gs held` on fresh worlds misses: `gs curved` is still recorded.
