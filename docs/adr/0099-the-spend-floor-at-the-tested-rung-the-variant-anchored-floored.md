# 0099. The spend floor at the tested rung: the variant `anchored floored`, never routed

Status: proposed, 2026-11-20; adds the `anchored` arm's variant `anchored floored`; changes no
default, no composite and no other variant's record.

Numbered 0097 on the branch where M4 was frozen: M4's pre-registration
(`scorecard/preregistration/m4.md` and `m4.json`, unchanged since the freeze) cites it as ADR 0097.

## Context

M3's claim arm at the tested rung was the composite `gs held` (ADR 0093): `anchored held`'s plan,
or `stacked`'s where that variant's record routes the world. Its claim held. On two exploratory
families, read on M2's worlds, its loss to PyMC-Marketing reached past the margin δ = 0.05 a euro:
family 1 (time-varying responses), +0.029 [+0.002, +0.055], and family 6 (truths outside the
history's range), +0.039 [+0.023, +0.055].

Read after the run, the route causes both losses. Without it, `anchored held` alone is within 0.05
of every rival on all 16 national families; its worst upper end is +0.045 (family 6), then +0.035
(family 13) and +0.032 (family 1). By the same reading `stacked` alone reaches +0.090 on family 1
and +0.069 on family 6. This reading was not declared before the run.

Family 6 is where `anchored held` comes closest to the margin. Its truths fall off below the range
of spend the history logged. The plan takes channels down there: 28 % to 31 % of `anchored held`'s
channel plans lie below the least of the history's adstock, against 3 % to 5 % of the best plan's
in the family's cliff environments.

A pre-registered laptop study on fresh seeds then looked for one change to `anchored held`,
unrouted. It declared 20 rules and 2 combinations of them before reading any rule's regret. The
rules were box cuts aimed at family 6 (raised floors, a cut to the adstock range), curve rules
aimed at family 1, and a shrinkage towards the status quo. It read each rule on M3's scored worlds
at the tested rung (3 480) and on 2 140 fresh worlds. The freezing rule was declared with it:

- *A no-loss bar.* On the concave families (0, 2, 3, 5, 7, 12, 13) the paired mean of the rule
  less `anchored held` is at most +0.002 a euro pooled and at most +0.010 in each family.
- *The smallest worst case.* Among the rules that pass, freeze the one of the smallest S. S is
  the largest, over families and rivals, of the predicted margin's upper end at 80 % power with
  520 worlds a family. A rule replaces `anchored held` only where its S lies below that variant's
  by more than 0.002.

| rule | S | S, read through `anchored held` | no-loss, pooled |
|---|---|---|---|
| `anchored held` | +0.0684 (family 6, PyMC-Marketing) | +0.0448 (family 6, PyMC-Marketing) | 0 |
| **the spend floor at 0.9** | **+0.0533** (family 6, PyMC-Marketing) | **+0.0348** (family 13, the equal split) | +0.0008 |
| the spend floor at 0.9 and the line where the tests' slope statistic is past 1.645 | +0.0554 | +0.0398 | +0.0014 |
| the curve posterior read off the tests alone | +0.0669 | +0.0433 | +0.0019 |
| the line where the tests' slope statistic is past 1.645 | +0.0729 | +0.0493 | +0.0006 |
| the line where the tests' slope statistic is past 1.0 | +0.0822 | +0.0586 | +0.0009 |

S's predicted margin is the rule less the status quo on fresh worlds plus the status quo less the
rival on M3's worlds. The third column reads it through `anchored held` instead: that variant less
the rival on M3's worlds plus the rule less that variant on fresh worlds. The study added that
reading before any rule was frozen. Every other rule failed the no-loss bar.

## Decision

- **The variant `anchored floored`** (`scorecard.arms.anchored`) is `anchored held` with the spend
  floor and no route. Four points define it.
  1. *The floor.* A channel the time guard leaves free gets the larger of its floor and
     `SPEND_FLOOR` = 0.9 times its least weekly spend over the export's whole history, at most its
     ceiling. The least is the weekly spend's, not the adstock's.
  2. *Floors that do not fit.* Where the floors so raised sum past the budget, the guard's box
     stands. A floor is never lowered: a free channel whose floor lies above its ceiling keeps it,
     and the box holds no plan, as in `anchored held`.
  3. *No route.* No composite reads the variant. Its record's `routed` is false on every world,
     and it keeps the route's two reasons as the fit read them.
  4. *Everything else is `anchored held`'s.* The fit, the curves, the guard and its three
     status-quo cases are that variant's. Where no floor is raised, the plan, the claimed gain and
     its interval are that variant's to the last bit.
- **The record.** Its flags add `floor applied`: whether a raised floor cut the box the plan was
  searched in. Its settings add the constant, `spend_floor`.
- **Why the route goes.** It is the source of both of M3's losses beyond δ at the tested rung, and
  the floor's study measured the rule unrouted. Without the route, the claim arm needs no fit of
  `stacked`: the arm's fit is a regression and its evidence.
- **What stays.** `anchored`, `anchored curved`, `anchored held` and the composites `gs`,
  `gs curved` and `gs held` keep their records. A root that declares the `anchored` arm now holds
  four variants' records. Defaults are unchanged.

## Consequences

- *The final read.* The study read the frozen rule once, beside `anchored held`, on a final half
  of 2 140 fresh worlds (seeds 825 000 to 829 439, every national family but 15). The floor less
  `anchored held`, paired, mean a euro with its 95 % interval:

  | family | the floor less `anchored held` |
  |---|---|
  | 6 | -0.0102 [-0.0179, -0.0025] |
  | 4 | +0.0123 [+0.0095, +0.0151] |
  | 19 | +0.0294 [+0.0158, +0.0430] |
  | 20 | +0.0038 [+0.0020, +0.0057] |
  | 0 | +0.0025 [+0.0019, +0.0031] |
  | 1 | +0.0011 [+0.0006, +0.0017] |
  | the other nine | between -0.0006 and +0.0009 |

  Against PyMC-Marketing on family 6, read through `anchored held`, the floor's predicted margin is
  +0.0199 [+0.0029, +0.0369]; `anchored held`'s is +0.0301 [+0.0149, +0.0453]. Read that way,
  every family is within 0.05 of every rival at the final read's point estimates. The most worlds
  a family needs for a verdict at 80 % power is 290 (family 1); `anchored held` needs 330 (family
  6). Read through the status quo, as first declared, both rules miss on family 14. There the
  rules' gain over the status quo is smaller on the fresh worlds than on M3's, by 0.073 a euro
  (the floor) and 0.076 (`anchored held`), and that reading adds the difference to the margin.
- *Where the floor costs.* Family 19 (a channel with a flat response), its environment
  `flat-pla`: +0.0604 [+0.0262, +0.0946] a euro over `anchored held` on the final half. The family
  stays ahead of PyMC-Marketing, -0.0233 [-0.0456, -0.0009]. Family 4 (S-curve truths): +0.0123.
  Family 15 (recorded siMMMulator worlds, read on M3's records alone): +0.025. In family 6's
  `saturation` environment, whose truths have no cliff, the floor also costs: +0.0154
  [+0.0038, +0.0270].
- *The constant is the study's.* 0.9 and 1.0 were the two shares declared, over three sources of
  the least: the weekly spend, the adstock over every week and the adstock past the kernel's
  ramp. Only the weekly spend at 0.9 passed the no-loss bar. No other share was read.
- *The port.* The study's 2 140 final-half worlds were planned again through this package's own
  path: the arm's records, scored by the bench. The variant's regret equals the study's rule to the
  last bit on 2 140 of 2 140 worlds, and `anchored held`'s equals the study's reading of that
  variant on every one. On every world the records of `anchored`, `anchored curved` and
  `anchored held` are those of the tree before this change, but for their cost. A raised floor cut
  the box on 1 432 worlds: every world of families 6 and 19, none of family 17, whose twins hold
  television at its status quo. Where none did, the variant's regret is `anchored held`'s exactly,
  on 708 of 708 worlds.
- *What no world tests.* A free channel's floor above its ceiling: no export of the bench has
  one. The study's form of the floor, `min(max(lower, 0.9 m), upper)`, lowers such a floor to the
  ceiling and plans; this variant keeps the floor and fails, as `anchored held` does. The two forms
  give the same bits wherever each floor lies at or below its ceiling.
- *Roots recorded before this change* hold no record of the variant. This tree reads every
  variant the `anchored` arm now has: the runner scores a world once each of them is recorded,
  and `scorecard.claim.read_beside` joins each of them from a root beside a run. Such a root is
  read with the tree that recorded it, as M3's are with M3's.
- *The cost.* The floor is one minimum a channel. The variant reads the curves in its own plan,
  as `anchored curved` and `anchored held` do, so a root that runs the arm reads them three times
  a world; ADR 0089 states what a read costs.
- *The tests* (`tests/scorecard/test_anchored_arm.py`), on the arm's eight fixture worlds. On the
  four whose every floor lies at or above the raised one, the record is `anchored held`'s to the
  last bit but for `routed`, which is false on the routed world too. On a world where
  `anchored held` plans meta at its box's floor, below 0.9 times its least weekly spend, the plan
  puts meta on the raised floor to the last bit: the water-filling's plan in the raised box. A
  held channel whose history never spent as little as its status quo stays there to the last bit,
  and the other two channels are planned in the raised box. Floors raised past the budget leave the
  guard's box: the plan, its claim and its interval are `anchored held`'s. A floor above its
  ceiling fails as `anchored held` fails. Each of 8 mutants of the variant turns at least one of
  these tests red; the floor removed turns the binding test red.

## Alternatives considered

- **Keep the route** (`gs held`): M3's two losses beyond δ.
- **A floor at the history's adstock minimum**, over every week or past the kernel's ramp, at
  0.9 or 1.0, and the floor at the least weekly spend itself (1.0): each failed the no-loss bar,
  pooled +0.0028 to +0.0112 a euro.
- **The box cut to the adstock range**, as the curved margin variant cuts it (ADR 0087): failed
  the no-loss bar, +0.0092 pooled.
- **Curve rules for family 1**: a channel planned on its line where its tests' slope or their trend
  in time is significant, the corner on the shrunk means where a trend is, the curve posterior off
  the tests alone, a heavier line prior. Those that passed the no-loss bar did not lower S below
  `anchored held`'s by more than 0.002.
- **The floor with a curve rule**: the floor with the line where the tests' slope statistic is past
  1.645 passed the bar with S +0.0554, above the floor alone.
