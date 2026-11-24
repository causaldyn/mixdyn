# Benchmark

How mixdyn's plans measure up against Meridian, PyMC-Marketing, Orbit, the status quo and the equal
split on the [causaldyn bench](https://github.com/causaldyn/causaldyn-bench): what three
pre-registered runs measured, where the engine loses, how each run can be reproduced, and what it
cost. Every table here is written by `scripts/readme_record.py` from the runs' records and every
chart by `scripts/record_charts.py`; `tests/test_readme.py` fails where either is not what its
script writes.

![mixdyn against its closest rival, family by family](assets/closest-rival-light.svg#gh-light-mode-only)
![mixdyn against its closest rival, family by family](assets/closest-rival-dark.svg#gh-dark-mode-only)

## Conflict of interest

One maintainer built both the engine and the bench that scores it: the bench's worlds, its regret
axis and the scripts that run each rival. The same maintainer wrote every run's pre-registration and
ran every run. No one independent of the engine has run any of them again. What limits the conflict:
each pre-registration was frozen before any confirmatory world was fitted, M4's and M5's with a
public time-stamp and M3's without one (below); each rival ran in its own pinned environment at its
recommended settings, and in M3 also at the setting its own tuning criterion chose; and the bench
and the harness are public. What does not limit it: the choice of the families, of the axis and of
the rivals' scripts.

## What has been measured

**The claim.** On the causaldyn bench's simulated worlds, whose truth is known, mixdyn's plans are
non-inferior to those of Meridian 2.1.0, PyMC-Marketing 1.3.0 and Orbit 1.1.5.1, and to the status
quo and the equal split, within δ = 0.05 of regret per euro of budget, on 37 pre-registered
family-rungs. Three scored runs hold them:

- **M4**: 26 verdicts on 15 national families, each tested at one-sided 0.025 in a fixed sequence,
  on fresh seeds that no development step and no earlier run drew. All 26 held; the
  pre-registration expected 25.23.
- **M5**, the run after it: 5 verdicts on what M4 left open, the k = 0 rungs of families 1, 6 and
  14 and both rungs of family 15 (siMMMulator), tested in the same way in a sequence of their own,
  on worlds of a third block of seeds that no earlier run drew (ADR 0101). All 5 held; the
  pre-registration expected 4.997.
- **M3**, the run before M4: the six verdicts of the geo families 8, 9 and 10, which neither later
  run ran. M3 tested them in the same way, in its own sequence of 17 verdicts, all of which held
  (16.94 expected). They stand.

Each row below names the run it comes from. Of the 32 national family-rungs, one is in no claim:
family 4 (S-curves) at k = 0, the one family-rung of the record not within δ. There `margin held`
loses to PyMC-Marketing by +0.1353 [+0.0997, +0.1708] and to Orbit by +0.1362 [+0.0997, +0.1727] a
euro on M4's worlds ("Where it loses"): with no lift test the data do not identify television's
curve beyond the range its spend was logged in. A later search for a k = 0 rule within δ on family 4
found none: its rule, frozen on M4's worlds and read once on held-out worlds, still trailed
PyMC-Marketing by +0.046 and Orbit by +0.054 a euro (M5's pre-registration, its Scope), so M5 did
not run the rung.

This is a non-inferiority claim. It does not say that mixdyn is the best MMM, and it does not say
that mixdyn is within δ of every maintained MMM (see "No headline" below).

- **The axis.** The regret of the quarter's budget split a tool recommends, per euro of budget: the
  KPI the optimal split earns and the recommended split does not. Lower is better.
- **A comparison** is mixdyn's arm less one rival, paired by world, over the worlds both planned.
  It is within δ where Hall's (1992) skewness-corrected one-look reader shows the mean below 0.05
  at one-sided 0.025. A verdict is one family at one rung: an intersection-union test over its
  comparisons, read at the largest of their levels.
- **The rungs.** k = 0 reads the history alone. The tested rung reads the history and four go-dark
  lift tests (family 7: one test on 52 weeks of history, two on 78 and 104 weeks).
- **The arms.** At k = 0, in every run, `margin held`: the README's margin rule, with its curves and
  its time guard. At the tested rung of M4 and M5, `anchored floored` (ADR 0099): margins anchored
  on the lift tests, planned along curves under the same time guard, as `anchored held` plans them
  (ADRs 0088, 0089 and 0093), with no route, and each channel the guard leaves free floored at 0.9
  times its least weekly spend in the history. On M3's geo families, `geo eb`, a closed-form model
  over geos fitted by empirical Bayes (ADR 0090).
- **The worlds.** M4 read each environment's confirmatory block of seeds past its first 300, which
  are M3's (ADR 0098): no earlier run and no development step read a world of M4. M5 read each
  environment's block of a third region, from seed 1 000 000 on (ADR 0101), and family 15's
  siMMMulator files written for those seeds: no earlier run read a world of M5.

Each cell is mixdyn's arm less the rival, in regret per euro: the mean over the worlds both
planned, with its paired 95 % Student-t interval, as the run's reading prints them. A negative cell
is a lower regret than the rival's. "Level" is the verdict's level in its run's sequence, which
tested it at 0.025. "Holm's loss" marks a comparison that Holm's step-down reads as a loss ("Where
it loses").

<!-- record:claim -->
| Family | Rung | Arm | Run | Worlds | Level | Status quo | Equal split | Meridian | PyMC-Marketing | Orbit |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 Track M v2 | 0 | `margin held` | M4 | 310 | 1e-12 | −0.1365 [−0.1547, −0.1183] | −0.3261 [−0.3598, −0.2924] | −0.0478 [−0.0647, −0.0309] | −0.1258 [−0.1544, −0.0971] | −0.2062 [−0.2400, −0.1724] |
| 0 Track M v2 | 4 | `anchored floored` | M4 | 310 | 1e-12 | −0.2428 [−0.2585, −0.2272] | −0.4324 [−0.4678, −0.3971] | −0.0677 [−0.0765, −0.0589] | +0.0096 [+0.0048, +0.0144], Holm's loss | −0.0479 [−0.0610, −0.0349] |
| 1 Drift | 0 | `margin held` | M5 | 240 | 1.09e-11 | −0.1460 [−0.1844, −0.1076] | −0.1450 [−0.1845, −0.1054] | −0.0797 [−0.1123, −0.0470] | −0.1260 [−0.1846, −0.0675] | −0.1780 [−0.2381, −0.1179] |
| 1 Drift | 4 | `anchored floored` | M4 | 522 | 3.86e-05 | −0.2940 [−0.3234, −0.2646] | −0.2601 [−0.2930, −0.2273] | −0.1195 [−0.1366, −0.1024] | +0.0090 [−0.0013, +0.0194] | −0.0489 [−0.0681, −0.0297] |
| 2 Spend confounded with demand | 0 | `margin held` | M4 | 270 | 1e-12 | −0.0849 [−0.1062, −0.0636] | −0.0610 [−0.0841, −0.0379] | −0.0503 [−0.0719, −0.0287] | −0.1448 [−0.1788, −0.1108] | −0.1274 [−0.1562, −0.0986] |
| 2 Spend confounded with demand | 4 | `anchored floored` | M4 | 270 | 1e-12 | −0.1726 [−0.1923, −0.1529] | −0.1487 [−0.1698, −0.1277] | −0.0865 [−0.0992, −0.0738] | +0.0017 [−0.0034, +0.0068] | −0.0710 [−0.0866, −0.0554] |
| 3 Collinear channels | 0 | `margin held` | M4 | 270 | 1e-12 | −0.0636 [−0.0822, −0.0451] | −0.0537 [−0.0723, −0.0351] | −0.0330 [−0.0485, −0.0174] | −0.0978 [−0.1262, −0.0694] | −0.1142 [−0.1461, −0.0823] |
| 3 Collinear channels | 4 | `anchored floored` | M4 | 270 | 4.82e-07 | −0.1721 [−0.1911, −0.1530] | −0.1621 [−0.1834, −0.1408] | −0.0792 [−0.0910, −0.0674] | +0.0101 [+0.0047, +0.0155], Holm's loss | −0.0608 [−0.0756, −0.0460] |
| 4 S-curves | 4 | `anchored floored` | M4 | 522 | 8.14e-07 | −0.3730 [−0.4075, −0.3386] | −0.5028 [−0.5484, −0.4573] | −0.0665 [−0.0863, −0.0467] | −0.0067 [−0.0178, +0.0044] | −0.0139 [−0.0410, +0.0133] |
| 5 Budget follows performance | 0 | `margin held` | M4 | 240 | 1e-12 | −0.0878 [−0.1109, −0.0647] | −0.0689 [−0.0916, −0.0462] | −0.0476 [−0.0677, −0.0275] | −0.1250 [−0.1607, −0.0892] | −0.1429 [−0.1811, −0.1046] |
| 5 Budget follows performance | 4 | `anchored floored` | M4 | 240 | 1e-12 | −0.1883 [−0.2084, −0.1683] | −0.1694 [−0.1941, −0.1447] | −0.0951 [−0.1081, −0.0821] | +0.0017 [−0.0023, +0.0056] | −0.0789 [−0.0951, −0.0627] |
| 6 Truth outside the observed range | 0 | `margin held` | M5 | 240 | 1e-12 | −0.1071 [−0.1358, −0.0783] | −0.1002 [−0.1241, −0.0763] | −0.0802 [−0.1063, −0.0542] | −0.2987 [−0.3561, −0.2412] | −0.3962 [−0.4579, −0.3344] |
| 6 Truth outside the observed range | 4 | `anchored floored` | M4 | 240 | 0.00117 | −0.1203 [−0.1464, −0.0941] | −0.0987 [−0.1236, −0.0739] | −0.0331 [−0.0522, −0.0141] | +0.0295 [+0.0163, +0.0427], Holm's loss | −0.0566 [−0.0805, −0.0328] |
| 7 Short series | 0 | `margin held` | M4 | 252 | 1e-12 | −0.0607 [−0.0806, −0.0408] | −0.0376 [−0.0580, −0.0173] | −0.0369 [−0.0567, −0.0171] | −0.0725 [−0.1037, −0.0412] | −0.1026 [−0.1357, −0.0696] |
| 7 Short series | 1–2 | `anchored floored` | M4 | 252 | 1e-12 | −0.1381 [−0.1588, −0.1175] | −0.1150 [−0.1395, −0.0905] | −0.0676 [−0.0826, −0.0526] | −0.0147 [−0.0293, −0.0001] | −0.0471 [−0.0638, −0.0304] |
| 12 Meridian's home ground | 0 | `margin held` | M4 | 460 | 1e-12 | −0.0451 [−0.0676, −0.0226] | −0.1516 [−0.1931, −0.1101] | −0.0164 [−0.0371, +0.0042] | −0.0834 [−0.1141, −0.0528] | −0.1696 [−0.2222, −0.1169] |
| 12 Meridian's home ground | 4 | `anchored floored` | M4 | 460 | 1e-12 | −0.1435 [−0.1717, −0.1154] | −0.2500 [−0.2959, −0.2041] | −0.0132 [−0.0198, −0.0067] | −0.0047 [−0.0100, +0.0005] | −0.0160 [−0.0241, −0.0079] |
| 13 PyMC-Marketing's home ground | 0 | `margin held` | M4 | 520 | 1e-12 | −0.0773 [−0.0935, −0.0612] | −0.0318 [−0.0467, −0.0170] | −0.0240 [−0.0371, −0.0110] | −0.2545 [−0.2814, −0.2275] | −0.3743 [−0.4053, −0.3434] |
| 13 PyMC-Marketing's home ground | 4 | `anchored floored` | M4 | 520 | 1e-12 | −0.0816 [−0.1008, −0.0624] | −0.0361 [−0.0562, −0.0159] | −0.0276 [−0.0418, −0.0134] | +0.0045 [−0.0041, +0.0132] | −0.1270 [−0.1464, −0.1077] |
| 14 Robyn's home ground | 0 | `margin held` | M5 | 240 | 1e-12 | −0.3648 [−0.4329, −0.2967] | −0.6511 [−0.7319, −0.5704] | −0.1446 [−0.2080, −0.0813] | −0.1627 [−0.2293, −0.0961] | −0.3298 [−0.4086, −0.2510] |
| 14 Robyn's home ground | 4 | `anchored floored` | M4 | 520 | 2.59e-07 | −0.3188 [−0.3785, −0.2591] | −0.5983 [−0.6677, −0.5288] | −0.0815 [−0.1171, −0.0458] | −0.0244 [−0.0506, +0.0018] | −0.1510 [−0.1934, −0.1086] |
| 15 siMMMulator | 0 | `margin held` | M5 | 240 | 1e-12 | −0.0201 [−0.0232, −0.0170] | −0.0632 [−0.0663, −0.0602] | −0.0296 [−0.0356, −0.0235] | +0.0236 [+0.0186, +0.0285], Holm's loss | −0.0263 [−0.0333, −0.0192] |
| 15 siMMMulator | 4 | `anchored floored` | M5 | 240 | 1e-12 | −0.0810 [−0.0836, −0.0784] | −0.1241 [−0.1267, −0.1216] | +0.0325 [+0.0296, +0.0353], Holm's loss | +0.0318 [+0.0295, +0.0340], Holm's loss | +0.0318 [+0.0292, +0.0343], Holm's loss |
| 17 Twins | 0 | `margin held` | M4 | 240 | 1e-12 | +0.0000 [+0.0000, +0.0000] | −0.1865 [−0.1931, −0.1799] | −0.4604 [−0.4899, −0.4309] | −0.0018 [−0.0040, +0.0005] | −0.1632 [−0.1899, −0.1365] |
| 17 Twins | 4 | `anchored floored` | M4 | 240 | 1e-12 | +0.0000 [+0.0000, +0.0000] | −0.1865 [−0.1931, −0.1799] | −0.0230 [−0.0281, −0.0178] | −0.0294 [−0.0373, −0.0215] | −0.0639 [−0.0761, −0.0517] |
| 18 Long term | 0 | `margin held` | M4 | 240 | 1e-12 | −0.1053 [−0.1277, −0.0828] | −0.0813 [−0.1037, −0.0589] | −0.0614 [−0.0817, −0.0410] | −0.1451 [−0.1834, −0.1068] | −0.0929 [−0.1229, −0.0629] |
| 18 Long term | 4 | `anchored floored` | M4 | 240 | 1e-12 | −0.1997 [−0.2214, −0.1780] | −0.1758 [−0.2001, −0.1514] | −0.0952 [−0.1085, −0.0819] | +0.0036 [−0.0023, +0.0094] | −0.0722 [−0.0881, −0.0562] |
| 19 Flat | 0 | `margin held` | M4 | 282 | 1.94e-09 | −0.0524 [−0.0672, −0.0376] | −0.0290 [−0.0502, −0.0078] | not compared: the rival failed 282 | −0.0269 [−0.0532, −0.0007] | −0.0680 [−0.1019, −0.0342] |
| 19 Flat | 4 | `anchored floored` | M4 | 282 | 1e-12 | −0.1504 [−0.1669, −0.1339] | −0.1269 [−0.1474, −0.1064] | not compared: the rival failed 282 | −0.0282 [−0.0475, −0.0089] | −0.0233 [−0.0377, −0.0090] |
| 20 Promotions | 0 | `margin held` | M4 | 240 | 1e-12 | −0.0787 [−0.1003, −0.0570] | −0.0346 [−0.0578, −0.0115] | −0.0312 [−0.0510, −0.0113] | −0.0778 [−0.1117, −0.0439] | −0.0587 [−0.0889, −0.0285] |
| 20 Promotions | 4 | `anchored floored` | M4 | 240 | 1e-12 | −0.2104 [−0.2312, −0.1896] | −0.1663 [−0.1867, −0.1460] | −0.1088 [−0.1217, −0.0959] | −0.0047 [−0.0109, +0.0015] | −0.0710 [−0.0874, −0.0546] |
| 8 Geo | 0 | `geo eb` | M3 | 240 | 1e-12 | −0.0766 [−0.0820, −0.0713] | −0.0412 [−0.0454, −0.0371] | −0.0429 [−0.0463, −0.0394] | −0.0078 [−0.0097, −0.0059] | −0.0675 [−0.0725, −0.0626] |
| 8 Geo | 4 | `geo eb` | M3 | 240 | 1e-12 | −0.0768 [−0.0821, −0.0715] | −0.0414 [−0.0456, −0.0373] | −0.0443 [−0.0477, −0.0409] | −0.0083 [−0.0102, −0.0065] | −0.0558 [−0.0602, −0.0515] |
| 9 Spillover between geos | 0 | `geo eb` | M3 | 240 | 1e-12 | −0.1122 [−0.1222, −0.1023] | −0.0532 [−0.0602, −0.0461] | −0.0479 [−0.0543, −0.0416] | +0.0001 [−0.0032, +0.0035] | −0.0567 [−0.0633, −0.0501] |
| 9 Spillover between geos | 4 | `geo eb` | M3 | 240 | 1e-12 | −0.1061 [−0.1164, −0.0958] | −0.0470 [−0.0541, −0.0399] | −0.0442 [−0.0505, −0.0379] | +0.0051 [+0.0016, +0.0087] | −0.0613 [−0.0707, −0.0519] |
| 10 Few geos | 0 | `geo eb` | M3 | 240 | 1e-12 | −0.0692 [−0.0765, −0.0619] | −0.0380 [−0.0433, −0.0327] | −0.0307 [−0.0355, −0.0259], exploratory | −0.0027 [−0.0052, −0.0003] | −0.0428 [−0.0486, −0.0369] |
| 10 Few geos | 4 | `geo eb` | M3 | 240 | 1e-12 | −0.0690 [−0.0763, −0.0618] | −0.0378 [−0.0432, −0.0324] | −0.0339 [−0.0389, −0.0289], exploratory | −0.0015 [−0.0041, +0.0011] | −0.0380 [−0.0437, −0.0322] |
<!-- /record:claim -->

"Worlds" counts the confirmatory worlds the family declared. A failed fit leaves its world out of
the comparisons that read it, and is counted ("Worlds per cell", below). No claim arm failed in M4
or M5. Meridian failed on every world of family 19, 282 at each rung: it refuses a channel whose
spend does not vary across time, which is the family's point, and M4's pre-registration expected it
to leave that family's claim. Orbit failed on 3 world-rungs of family 18. In M3, PyMC-Marketing
failed on 41 world-rungs of the geo families, and in M5 on 22 world-rungs of family 15, 18 at k = 0
and 4 at the tested rung. On family 10, M3 declared Meridian capped, as in the run before it, so its
comparisons there are exploratory: it ran on every world, and it is read beside that verdict, not
in it.

**The error bars.** Each interval is the paired 95 % Student-t interval of the mean difference over
a cell's complete cases: `mean ± t(0.975, n − 1) · s / √n`, where `n` is the number of complete
cases and `s` the standard deviation of the per-world differences, with divisor `n − 1`. The
verdict's word and level are Hall's reader's; the interval printed is Student's. The variability
is from world to world. A world is the bench's draw from one seed of one of the family's
environments, which draws what the environment leaves free: on Track M v2's `reference` worlds,
whose truth is fixed, the data and the noise; on its `drawn` worlds the channels' curves too. A
family's environments are pooled into one sample. Each arm made one plan for each world and rung,
so the Monte Carlo error of a sampled fit is inside the spread of the differences: no world was
fitted again with other seeds to separate it. The intervals say nothing about worlds outside the
families.

**Worlds per cell.** The complete cases each cell above is read on: the worlds where both the arm
and the rival made a plan, from each run's reading (`scripts/readme_record.py` writes the table).

<!-- record:cells -->
| Family | Rung | Arm | Run | Status quo | Equal split | Meridian | PyMC-Marketing | Orbit |
|---|---|---|---|---|---|---|---|---|
| 0 Track M v2 | 0 | `margin held` | M4 | 310 | 310 | 310 | 310 | 310 |
| 0 Track M v2 | 4 | `anchored floored` | M4 | 310 | 310 | 310 | 310 | 310 |
| 1 Drift | 0 | `margin held` | M5 | 240 | 240 | 240 | 240 | 240 |
| 1 Drift | 4 | `anchored floored` | M4 | 522 | 522 | 522 | 522 | 522 |
| 2 Spend confounded with demand | 0 | `margin held` | M4 | 270 | 270 | 270 | 270 | 270 |
| 2 Spend confounded with demand | 4 | `anchored floored` | M4 | 270 | 270 | 270 | 270 | 270 |
| 3 Collinear channels | 0 | `margin held` | M4 | 270 | 270 | 270 | 270 | 270 |
| 3 Collinear channels | 4 | `anchored floored` | M4 | 270 | 270 | 270 | 270 | 270 |
| 4 S-curves | 4 | `anchored floored` | M4 | 522 | 522 | 522 | 522 | 522 |
| 5 Budget follows performance | 0 | `margin held` | M4 | 240 | 240 | 240 | 240 | 240 |
| 5 Budget follows performance | 4 | `anchored floored` | M4 | 240 | 240 | 240 | 240 | 240 |
| 6 Truth outside the observed range | 0 | `margin held` | M5 | 240 | 240 | 240 | 240 | 240 |
| 6 Truth outside the observed range | 4 | `anchored floored` | M4 | 240 | 240 | 240 | 240 | 240 |
| 7 Short series | 0 | `margin held` | M4 | 252 | 252 | 252 | 252 | 252 |
| 7 Short series | 1–2 | `anchored floored` | M4 | 252 | 252 | 252 | 252 | 252 |
| 12 Meridian's home ground | 0 | `margin held` | M4 | 460 | 460 | 460 | 460 | 460 |
| 12 Meridian's home ground | 4 | `anchored floored` | M4 | 460 | 460 | 460 | 460 | 460 |
| 13 PyMC-Marketing's home ground | 0 | `margin held` | M4 | 520 | 520 | 520 | 520 | 520 |
| 13 PyMC-Marketing's home ground | 4 | `anchored floored` | M4 | 520 | 520 | 520 | 520 | 520 |
| 14 Robyn's home ground | 0 | `margin held` | M5 | 240 | 240 | 240 | 240 | 240 |
| 14 Robyn's home ground | 4 | `anchored floored` | M4 | 520 | 520 | 520 | 520 | 520 |
| 15 siMMMulator | 0 | `margin held` | M5 | 240 | 240 | 240 | 222 (the rival failed 18) | 240 |
| 15 siMMMulator | 4 | `anchored floored` | M5 | 240 | 240 | 240 | 236 (the rival failed 4) | 240 |
| 17 Twins | 0 | `margin held` | M4 | 240 | 240 | 240 | 240 | 240 |
| 17 Twins | 4 | `anchored floored` | M4 | 240 | 240 | 240 | 240 | 240 |
| 18 Long term | 0 | `margin held` | M4 | 240 | 240 | 240 | 240 | 239 (the rival failed 1) |
| 18 Long term | 4 | `anchored floored` | M4 | 240 | 240 | 240 | 240 | 238 (the rival failed 2) |
| 19 Flat | 0 | `margin held` | M4 | 282 | 282 | 0 (the rival failed 282) | 282 | 282 |
| 19 Flat | 4 | `anchored floored` | M4 | 282 | 282 | 0 (the rival failed 282) | 282 | 282 |
| 20 Promotions | 0 | `margin held` | M4 | 240 | 240 | 240 | 240 | 240 |
| 20 Promotions | 4 | `anchored floored` | M4 | 240 | 240 | 240 | 240 | 240 |
| 8 Geo | 0 | `geo eb` | M3 | 240 | 240 | 240 | 235 (the rival failed 5) | 240 |
| 8 Geo | 4 | `geo eb` | M3 | 240 | 240 | 240 | 234 (the rival failed 6) | 240 |
| 9 Spillover between geos | 0 | `geo eb` | M3 | 240 | 240 | 240 | 233 (the rival failed 7) | 240 |
| 9 Spillover between geos | 4 | `geo eb` | M3 | 240 | 240 | 240 | 234 (the rival failed 6) | 240 |
| 10 Few geos | 0 | `geo eb` | M3 | 240 | 240 | 240 (exploratory) | 236 (the rival failed 4) | 240 |
| 10 Few geos | 4 | `geo eb` | M3 | 240 | 240 | 240 (exploratory) | 227 (the rival failed 13) | 240 |
<!-- /record:cells -->

**Hall's reader's words.** How often the claim's arm `beats` each rival, `ties` it or `loses to`
it over the claim's verdicts, by Hall's reader's word, unadjusted. A word is not a verdict: a
verdict reads only whether each comparison is within δ.

<!-- record:words -->
| Run | Arm | Rung | Verdicts | Status quo | Equal split | Meridian | PyMC-Marketing | Orbit |
|---|---|---|---|---|---|---|---|---|
| M4 | `margin held` | 0 | 11 | beats 10, ties 1 | beats 11 | beats 9, ties 1, not compared 1 | beats 10, ties 1 | beats 11 |
| M4 | `anchored floored` | the tested rung | 15 | beats 14, ties 1 | beats 15 | beats 14, not compared 1 | beats 3, ties 8, loses to 4 | beats 14, ties 1 |
| M5 | `margin held` | 0 | 4 | beats 4 | beats 4 | beats 4 | beats 3, loses to 1 | beats 4 |
| M5 | `anchored floored` | the tested rung | 1 | beats 1 | beats 1 | loses to 1 | loses to 1 | loses to 1 |
| M3 | `geo eb` | 0 | 3 | beats 3 | beats 3 | beats 3 | beats 2, ties 1 | beats 3 |
| M3 | `geo eb` | the tested rung | 3 | beats 3 | beats 3 | beats 3 | beats 1, ties 1, loses to 1 | beats 3 |
<!-- /record:words -->

- **k = 0.** `margin held` has a lower regret than the equal split and Orbit on all of M4's 11
  confirmatory families, and than Meridian on all but two: it ties Meridian on family 12, and
  Meridian is not compared on family 19. On family 17 its time guard plays the status quo, as
  declared, and it ties PyMC-Marketing there.
- **The tested rung.** `anchored floored` has a lower regret than the equal split on all 15
  families, than Meridian wherever Meridian is compared, and than Orbit and the status quo on all
  but one family each: it ties Orbit on family 4 and plays the status quo on family 17. Against
  PyMC-Marketing it is level: within δ on all 15 families, it beats it on 3, ties it on 8 and loses
  to it on 4, every loss inside δ ("Where it loses").
- **M5.** At k = 0 `margin held` has a lower regret than every rival on families 1, 6 and 14, and
  on family 15 than every rival but PyMC-Marketing, to which it loses by
  +0.0236 [+0.0186, +0.0285], inside δ. At family 15's tested rung `anchored floored` has a lower
  regret than the status quo and the equal split and loses to Meridian, PyMC-Marketing and Orbit,
  each loss inside δ ("Where it loses").
- **The geo families (M3).** `geo eb` has a lower regret than Meridian, Orbit, the status quo and
  the equal split on families 8, 9 and 10. Against PyMC-Marketing it is level.

## Where it loses

These were read in the same runs. They are as much a part of the result as the claim above. The
chart puts every comparison of the claim, and of family 4 at k = 0, on one axis, rival by rival.

![Every comparison, rival by rival](assets/every-comparison-light.svg#gh-light-mode-only)
![Every comparison, rival by rival](assets/every-comparison-dark.svg#gh-dark-mode-only)

Each comparison below is one of the claim's, or of M4's rungs that were not claimed, that Hall's
reader reads as a loss, that is not within δ, that Holm's step-down reads as a loss, or that the
secondary reading at δ = 0.01 does not find within; that reading covers the claimed comparisons
only ("not read").

<!-- record:losses -->
| Family | Rung | Arm | Run | Rival | Mean [95 % interval] | Hall's word | Within δ = 0.05 | Holm's loss | Within δ = 0.01 |
|---|---|---|---|---|---|---|---|---|---|
| 0 Track M v2 | 4 | `anchored floored` | M4 | PyMC-Marketing | +0.0096 [+0.0048, +0.0144] | loses to | yes | yes | no |
| 1 Drift | 4 | `anchored floored` | M4 | PyMC-Marketing | +0.0090 [−0.0013, +0.0194] | loses to | yes | no | no |
| 3 Collinear channels | 4 | `anchored floored` | M4 | PyMC-Marketing | +0.0101 [+0.0047, +0.0155] | loses to | yes | yes | no |
| 4 S-curves | 0 | `margin held` | M4 | PyMC-Marketing | +0.1353 [+0.0997, +0.1708] | loses to | no | yes | not read |
| 4 S-curves | 0 | `margin held` | M4 | Orbit | +0.1362 [+0.0997, +0.1727] | loses to | no | yes | not read |
| 4 S-curves | 4 | `anchored floored` | M4 | Orbit | −0.0139 [−0.0410, +0.0133] | ties | yes | no | no |
| 6 Truth outside the observed range | 4 | `anchored floored` | M4 | PyMC-Marketing | +0.0295 [+0.0163, +0.0427] | loses to | yes | yes | no |
| 13 PyMC-Marketing's home ground | 4 | `anchored floored` | M4 | PyMC-Marketing | +0.0045 [−0.0041, +0.0132] | ties | yes | no | no |
| 15 siMMMulator | 0 | `margin held` | M5 | PyMC-Marketing | +0.0236 [+0.0186, +0.0285] | loses to | yes | yes | no |
| 15 siMMMulator | 4 | `anchored floored` | M5 | Meridian | +0.0325 [+0.0296, +0.0353] | loses to | yes | yes | no |
| 15 siMMMulator | 4 | `anchored floored` | M5 | PyMC-Marketing | +0.0318 [+0.0295, +0.0340] | loses to | yes | yes | no |
| 15 siMMMulator | 4 | `anchored floored` | M5 | Orbit | +0.0318 [+0.0292, +0.0343] | loses to | yes | yes | no |
| 9 Spillover between geos | 4 | `geo eb` | M3 | PyMC-Marketing | +0.0051 [+0.0016, +0.0087] | loses to | yes | no | yes |
<!-- /record:losses -->

- **S-curves with no lift test (family 4, k = 0): not within δ, as predicted.** `margin held` loses
  to PyMC-Marketing by +0.1353 [+0.0997, +0.1708] and to Orbit by +0.1362 [+0.0997, +0.1727] a
  euro, the largest losses of M4's claim arms. M4's pre-registration predicted both from M3's
  records, where they read +0.184 and +0.171 on the worlds of the run before M3, and left them out
  of its claim (its Predicted failures); the rung was past the sizing cap in any case. At the
  tested rung, `anchored floored` is within δ of every rival on the same family. The curves are
  what lose: the recorded variants without them, plain `margin` and `margin guarded`, have a lower
  regret than PyMC-Marketing there and tie Orbit, while `margin curved` loses to both (the
  reading's recorded arms, unadjusted).
- **M4's Holm's losses within δ: PyMC-Marketing at the tested rung on families 0, 3 and 6.** Holm's
  step-down at familywise 0.025 over 148 comparisons finds five losses: family 4's two at k = 0,
  above, and these three. They lie inside δ, so the verdicts hold, but they are losses. Family 6's,
  +0.0295 [+0.0163, +0.0427], is in the claim's closest verdict, at level 0.00117. The 148 count the
  20 comparisons of the k = 0 rungs that were not claimed ("M4's deviations", below); over the 128
  claimed comparisons, Holm's losses are the same three.
- **At δ = 0.01**, a secondary reading, 122 of M4's 128 claimed comparisons are within. The six
  that are not are all at the tested rung: PyMC-Marketing on families 0, 1, 3, 6 and 13, and Orbit
  on family 4. Family 1's, +0.0090 [−0.0013, +0.0194], is a loss by Hall's reader's word (level
  0.0228, unadjusted), though its Student interval holds 0, and not one of Holm's. In M3's geo
  verdicts every comparison is within 0.01; on family 9 at the tested rung `geo eb` loses to
  PyMC-Marketing by +0.0051 [+0.0016, +0.0087], unadjusted.
- **M5's Holm's losses within δ: all four on family 15 (siMMMulator).** Holm's step-down at
  familywise 0.025 over M5's 25 comparisons finds four losses: PyMC-Marketing at k = 0,
  +0.0236 [+0.0186, +0.0285], and at the tested rung Meridian, +0.0325 [+0.0296, +0.0353],
  PyMC-Marketing, +0.0318 [+0.0295, +0.0340], and Orbit, +0.0318 [+0.0292, +0.0343]. They lie
  inside δ, so both verdicts hold, but they are losses, and the same four are the only ones of
  M5's 25 claimed comparisons not within 0.01. At the tested rung the recorded variant `anchored`,
  with neither the curves nor the floor, ties Meridian and has a lower regret than PyMC-Marketing
  and Orbit; with the curves (`anchored curved`, `anchored held`) it loses to all three by less
  than 0.007 a euro, and the floor adds the rest (the reading's recorded arms, unadjusted).
- **M3's losses beyond δ at the tested rung do not recur.** In M3, read on the worlds of the run
  before it, `gs held` lost to PyMC-Marketing on family 1 by +0.0286 [+0.0019, +0.0553] and on
  family 6 by +0.0386 [+0.0225, +0.0548]. On M4's fresh worlds `anchored floored` is within δ of it
  on both (the claim's table).
- **M4's k = 0 rungs that were not claimed.** On families 1, 4, 6 and 14 the sizing rule put k = 0
  past the cap of 520 worlds (M4's pre-registration, Sizing): run and read on the family's worlds,
  never claimed in M4. On families 1, 6 and 14, `margin held` has a lower regret than every rival
  there, each comparison within δ, and M5 claimed all three on fresh worlds (the claim's table);
  family 4's losses are above.

<!-- record:exploratory -->
| Family | Rung | Arm | Worlds | Status quo | Equal split | Meridian | PyMC-Marketing | Orbit |
|---|---|---|---|---|---|---|---|---|
| 1 Drift | 0 | `margin held` | 522 | −0.1699 [−0.1960, −0.1438], beats | −0.1360 [−0.1628, −0.1092], beats | −0.0967 [−0.1192, −0.0741], beats | −0.1700 [−0.2084, −0.1317], beats | −0.2070 [−0.2498, −0.1642], beats |
| 4 S-curves | 0 | `margin held` | 522 | −0.2453 [−0.2729, −0.2177], beats | −0.3751 [−0.4100, −0.3403], beats | −0.0929 [−0.1199, −0.0658], beats | +0.1353 [+0.0997, +0.1708], loses to, not within δ, Holm's loss | +0.1362 [+0.0997, +0.1727], loses to, not within δ, Holm's loss |
| 6 Truth outside the observed range | 0 | `margin held` | 240 | −0.1084 [−0.1325, −0.0844], beats | −0.0869 [−0.1065, −0.0673], beats | −0.0666 [−0.0891, −0.0440], beats | −0.2938 [−0.3448, −0.2428], beats | −0.4037 [−0.4618, −0.3456], beats |
| 14 Robyn's home ground | 0 | `margin held` | 520 | −0.4602 [−0.5191, −0.4013], beats | −0.7397 [−0.8065, −0.6728], beats | −0.1287 [−0.1704, −0.0869], beats | −0.1159 [−0.1598, −0.0721], beats | −0.3183 [−0.3754, −0.2613], beats |
<!-- /record:exploratory -->

- **No headline.** "Within δ of every maintained MMM" needs every family and rung claimed whole and
  every verdict held. Over the record it cannot be read: family 4 at k = 0 is in no claim, and its
  losses are beyond δ. M4's pre-registration said in advance that M4's headline could not be
  read, and M5's reading prints its headline as holding over its own five verdicts only.
- **Not covered by fresh worlds since M3.** The geo verdicts rest on M3's run, which neither later
  run repeated ("What the runs did not cover", below).

## M4's deviations

What a reader of this record needs from the run's deviations from its pre-registration:

- **The k = 0 rungs past the cap print as "predicted to fail".** The run's roots kept the k = 0
  rungs of families 1, 4, 6 and 14 out of the fixed sequence by naming every rival there as
  predicted to fail. So the reading prints `not claimed: predicted to fail` on all 20 of their
  comparisons, and Holm's list counts them: 148 comparisons, the 128 claimed and these 20. The only
  predicted failures in the pre-registration are family 4's two at k = 0, against PyMC-Marketing
  and Orbit; the other 18 are exploratory. Over the 128 claimed comparisons, Holm's losses are the
  same three within δ. No verdict reads the field. A run can now declare such rungs exploratory
  (`Declaration.exploratory_rungs`, ADR 0100); M4's roots declare `predicted`, so its reading is
  unchanged.
- **The reading left out the tested rung's recorded variants.** `claim read --recorded` at the
  run's commit printed each recorded arm of a run's own root less every rival, but not the
  variants of its joined root, `anchored`, `anchored curved` and `anchored held`. The reader reads
  them since commit `4d89ee2`: read again there, no line of M4's reading moves and 225 lines of
  these variants are added. They are recorded beside the claim and enter no verdict.
- **Family 17's twins read the same seeds.** The pre-registration's seed formula gives each
  environment a block of its own, but the bench draws the twins, `curve` and `line`, from the same
  seeds by design: both read the same 120 seeds, and the reading pools the 240 worlds as if they
  were independent.
- **Two spot interruptions.** The box was launched three times, and each launch resumed the stage
  (Compute, below).

## The tuned rivals

M4 and M5 ran no tuned rival and no Robyn (M4's pre-registration's Deviations, items 2 and 3, and
M5's, item 6). This section is M3's, whose arms were `margin held` at k = 0 and, at the tested rung,
`gs held`: it plays `anchored held` and routes a world that `anchored held` flags to the Bayesian
engine's `stacked` plan (ADRs 0088, 0089 and 0093); on the geo families, `geo eb`.

Each rival ran at its recommended settings. Beside two of them ran the setting that the tool's own
criterion chose on the pilot's worlds, before any confirmatory world and never by regret
(`scorecard/tuned.json`; M3's pre-registration's Arms): `meridian tuned`, Meridian with
`contribution` priors and knots at most 13 weeks apart, chosen by its health card and its
out-of-sample error; `robyn tuned`, Robyn with `ts_validation` on. PyMC-Marketing's tuning chose
its recommended setting, so it has no tuned column. No development record holds a tuned setting,
so neither tuned rival has a predictive: both are read on their roots' worlds, exploratory, and
enter no verdict. Robyn ran on 20 worlds an environment of families 9, 12 and 13.

Each cell is the claim's arm less the rival, regret per euro, with its paired 95 % Student-t
interval and the reader's word, as M3's reading prints them; a Robyn cell names its worlds.
The word is Hall's reader's, which corrects for the differences' skew: on a skewed sample it can
read `beats` where the Student interval crosses 0.

<!-- record:tuned -->
| Family | Rung | Arm | Meridian | Meridian tuned | Robyn | Robyn tuned |
|---|---|---|---|---|---|---|
| 0 Track M v2 | 0 | `margin held` | −0.0515 [−0.0722, −0.0309], beats | −0.1206 [−0.1392, −0.1019], beats | not run | not run |
| 0 Track M v2 | 4 | `gs held` | −0.0764 [−0.0853, −0.0676], beats | −0.0737 [−0.0839, −0.0635], beats | not run | not run |
| 2 Spend confounded with demand | 0 | `margin held` | −0.0437 [−0.0688, −0.0186], beats | −0.0700 [−0.0927, −0.0473], beats | not run | not run |
| 2 Spend confounded with demand | 4 | `gs held` | −0.0680 [−0.0817, −0.0542], beats | −0.0840 [−0.1006, −0.0673], beats | not run | not run |
| 3 Collinear channels | 0 | `margin held` | −0.0211 [−0.0402, −0.0020], beats | −0.0420 [−0.0622, −0.0218], beats | not run | not run |
| 3 Collinear channels | 4 | `gs held` | −0.0816 [−0.0934, −0.0697], beats | −0.0936 [−0.1081, −0.0792], beats | not run | not run |
| 5 Budget follows performance | 0 | `margin held` | −0.0421 [−0.0617, −0.0225], beats | −0.0582 [−0.0781, −0.0383], beats | not run | not run |
| 5 Budget follows performance | 4 | `gs held` | −0.0690 [−0.0834, −0.0545], beats | −0.0820 [−0.0988, −0.0653], beats | not run | not run |
| 7 Short series | 0 | `margin held` | −0.0453 [−0.0664, −0.0242], beats | −0.0528 [−0.0714, −0.0342], beats | not run | not run |
| 7 Short series | 1–2 | `gs held` | −0.0616 [−0.0776, −0.0457], beats | −0.0739 [−0.0923, −0.0555], beats | not run | not run |
| 8 Geo | 0 | `geo eb` | −0.0429 [−0.0463, −0.0394], beats | −0.0556 [−0.0603, −0.0509], beats | not run | not run |
| 8 Geo | 4 | `geo eb` | −0.0443 [−0.0477, −0.0409], beats | −0.0554 [−0.0600, −0.0508], beats | not run | not run |
| 9 Spillover between geos | 0 | `geo eb` | −0.0479 [−0.0543, −0.0416], beats | −0.0665 [−0.0739, −0.0590], beats | −0.0356 [−0.0527, −0.0186], beats, 40 worlds | −0.0401 [−0.0591, −0.0211], beats, 40 worlds |
| 9 Spillover between geos | 4 | `geo eb` | −0.0442 [−0.0505, −0.0379], beats | −0.0687 [−0.0779, −0.0595], beats | −0.0741 [−0.0980, −0.0502], beats, 40 worlds | −0.0873 [−0.1177, −0.0570], beats, 40 worlds |
| 10 Few geos | 0 | `geo eb` | −0.0307 [−0.0355, −0.0259], beats | −0.0335 [−0.0387, −0.0284], beats | not run | not run |
| 10 Few geos | 4 | `geo eb` | −0.0339 [−0.0389, −0.0289], beats | −0.0370 [−0.0423, −0.0317], beats | not run | not run |
| 12 Meridian's home ground | 0 | `margin held` | −0.0216 [−0.0445, +0.0013], beats | −0.0338 [−0.0599, −0.0078], beats | −0.1368 [−0.3813, +0.1078], ties, 20 worlds | −0.2104 [−0.4753, +0.0545], beats, 20 worlds |
| 12 Meridian's home ground | 4 | `gs held` | −0.0128 [−0.0226, −0.0031], beats | −0.0358 [−0.0536, −0.0180], beats | −0.3061 [−0.5726, −0.0396], beats, 20 worlds | −0.3341 [−0.6172, −0.0509], beats, 20 worlds |
| 13 PyMC-Marketing's home ground | 0 | `margin held` | −0.0102 [−0.0299, +0.0094], ties | −0.0618 [−0.0823, −0.0413], beats | −0.1850 [−0.2764, −0.0937], beats, 20 worlds | −0.2928 [−0.4187, −0.1670], beats, 20 worlds |
| 13 PyMC-Marketing's home ground | 4 | `gs held` | −0.0250 [−0.0458, −0.0043], beats | −0.0916 [−0.1162, −0.0670], beats | −0.1025 [−0.2471, +0.0421], ties, 20 worlds | −0.1686 [−0.3428, +0.0055], beats, 20 worlds |
| 17 Twins | 0 | `margin held` | −0.4751 [−0.5188, −0.4313], beats | −0.6714 [−0.6808, −0.6619], beats | not run | not run |
| 17 Twins | 4 | `gs held` | −0.0224 [−0.0300, −0.0148], beats | −0.0634 [−0.0730, −0.0537], beats | not run | not run |
| 18 Long term | 0 | `margin held` | −0.0348 [−0.0606, −0.0089], beats | −0.0604 [−0.0856, −0.0352], beats | not run | not run |
| 18 Long term | 4 | `gs held` | −0.0730 [−0.0914, −0.0547], beats | −0.0874 [−0.1097, −0.0651], beats | not run | not run |
| 19 Flat | 0 | `margin held` | not compared: the rival failed 180 | not compared: the rival failed 180 | not run | not run |
| 19 Flat | 4 | `gs held` | not compared: the rival failed 180 | not compared: the rival failed 180 | not run | not run |
| 20 Promotions | 0 | `margin held` | −0.0246 [−0.0455, −0.0036], beats | −0.0434 [−0.0768, −0.0100], beats | not run | not run |
| 20 Promotions | 4 | `gs held` | −0.0842 [−0.1007, −0.0676], beats | −0.0656 [−0.0806, −0.0505], beats | not run | not run |

Over these rows the claim's arm reads: against Meridian it beats on 25, ties on 1, is not compared
on 2, of 28; against Meridian tuned it beats on 26, is not compared on 2, of 28; against Robyn it
beats on 4, ties on 2, of 6; against Robyn tuned it beats on 6, of 6.
<!-- /record:tuned -->

## What the runs did not cover

- **Bench families not scored.** Family 11 (Google's AMSS simulator) was withdrawn. Family 16, a
  published third-party instance of endogenous spend chosen in its place, was in neither run: its
  files and their licence had not arrived. Family 21 (mmm-framework's own worlds) was in neither
  run: its truth states each channel's contribution and return and no response curve, so it scores
  a model's reading, not a plan's regret. Family 10's 50-geo environment was not run: its only
  reading is an audit on fresh worlds before M3. Family 1's `cpm` environment and family 2's
  `search-demand` environment were in neither run's declaration.
- **Family 15 (siMMMulator 1.1.0) was not run in M4.** Its worlds are files siMMMulator writes, not
  draws, so M4 could not read a world of it that no run had read. M3 read it, exploratory, on the
  worlds of the run before it. M4's pre-registration reports, beside its claim, a development
  reading of the tested rung's rule on family 15's 100 recorded worlds (its Reported regardless).
  M5 read it on 240 files written for its own seeds before its pre-registration was frozen, whose
  digests the pinned bench holds.
- **Family 4 at k = 0 was not run in M5** (What has been measured, above): M4's reading of it
  stands, with its losses beyond δ.
- **The geo families (8, 9 and 10) were not run in M4 or M5.** Their verdicts are M3's, on M3's
  fresh worlds, and no later run has read them.
- **Robyn 3.12.1 and the tuned rivals were not run in M4 or M5.** M3 read them beside its claim,
  never in it ("The tuned rivals", above).
- **PyMC-Marketing ran its static model on every family**: logistic saturation and a static
  intercept, as pre-registered, with no time-varying term. Its results on the drift family (1) are
  that model's.
- **The tested rung's claim arm was chosen after M3's results**, by a research round on M3's
  records and on worlds drawn for the purpose, as M4's pre-registration records (its Deviations,
  item 11). M4's worlds are fresh, so that choice could not select its verdicts, but it could make
  their predictive optimistic. On the fresh worlds 26 of 26 held, against 25.23 expected.
- **Simulated worlds only.** No claim here is about real data (below).

## Pre-registration and reproduction

- **M4.** Its pre-registration is `scorecard/preregistration/m4.md`, with `m4.json` and
  `m4_sizing.py` beside it, byte for byte as frozen. It was frozen on 2026-11-21 in UTC+2, late on
  2026-11-20 in UTC, before any confirmatory world was fitted, in this project's then-private
  repository. Its SHA-256 is
  `ec0df163332e40244ba0747ae1c6c9654664fc7354137a9498ad5d979e9e45b1`, which
  `sha256sum scorecard/preregistration/m4.md` prints. Where it cites ADRs 0096 and 0097, this
  repository numbers them 0098 and 0099.
- **M4's public time-stamp.** The public bench's SSH-signed tag
  [`preregistration-m4`](https://github.com/causaldyn/causaldyn-bench/tree/preregistration-m4),
  on its commit `5ddfe0c`, names the SHA-256 of `m4.md`, `m4.json` and `m4_sizing.py` and the
  mixdyn commit the run ran (`git cat-file tag preregistration-m4` in a clone of the bench prints
  them). The tag is dated 2026-11-20 23:52:29 UTC, and GitHub's record of its signature check is
  dated 2026-11-20 23:52:55 UTC: before the run's box was first launched, at 06:27:48 UTC on
  2026-11-21. GitHub could check the tag only once it held it.
- **M4's code.** The run ran mixdyn `cc83d2a`, a commit of the then-private repository that is not
  public by that id. Its `src/` tree, `782ca30c324a5720c37c4b4e6856862d4f22a2f5`, is this
  repository's at v0.1.0, which M3 ran; its `scorecard/` tree is
  `d9149edd724b4efc8dc326f8715a43ceeb925709`. This repository's commit `b71572e` adds M4's
  pre-registration and the harness it ran on; its trees, `ac14998` for `src/` and `1a50f05` for
  `scorecard/`, are main's at `631c51b` too. They differ from the run's in these files only:
  - `src/mixdyn/inference/sensitivity.py`, `src/mixdyn/models/bayes_mmm.py` and
    `src/mixdyn/transforms/saturation.py`: the `tanh` curve's constant `atanh(1/2)` is a
    correctly rounded literal (commit `5cee2ad`, ADR 0096);
  - `src/mixdyn/models/margin_mmm.py`: the margin regression reads under the noise mixture by
    default, and the mixture's memory is bounded (commits `65d9586` and `486d83a`, ADR 0097);
  - `scorecard/arms/margin.py` and `scorecard/arms/anchored.py`: both pass `noise=None` to the
    margin regression, the regression the run read, and their docstrings say so; `anchored.py`
    cites `anchored floored`'s ADR as 0099, which the run's tree numbered 0097.

  `git diff v0.1.0 b71572e -- src` prints the first two; `git log b71572e.. -- src scorecard`
  lists every later change. The run locked causal-hybrid-control 1.3.1 and jax 0.12.0.
- **M5.** Its pre-registration is `scorecard/preregistration/m5.md`, with `m5.json` and
  `m5_sizing.py` beside it, byte for byte as frozen. It was frozen on 2026-11-23 at 16:31 UTC,
  before any confirmatory world was fitted, in the project's private development repository. Its
  SHA-256 is `86e12d7cc1e2b8eae21689db069c5600f0fc8a9d64d797c2a9350ac2ba7970d1`, which
  `sha256sum scorecard/preregistration/m5.md` prints. Where it cites ADR 0097 for
  `anchored floored`, the run's tree and this repository number that ADR 0099.
- **M5's public time-stamp.** The public bench's SSH-signed tag
  [`preregistration-m5`](https://github.com/causaldyn/causaldyn-bench/tree/preregistration-m5),
  on its commit `bc52170` (branch `m5`), names the SHA-256 of `m5.md`, `m5.json` and
  `m5_sizing.py` and the mixdyn commit the run ran (`git cat-file tag preregistration-m5` prints
  them). The tag is dated 2026-11-23 16:31:05 UTC, and GitHub's public events of the bench record
  the branch `m5`, which holds the tagged commit, as created at 16:31:06 UTC: before the run's box
  was launched, at 17:05:36 UTC the same day. GitHub's record of the tag's signature check is dated
  after the run, so it does not date the tag.
- **M5's code.** The run ran mixdyn `b864f6d`, a commit of the private repository that is not
  public by that id. This repository's commit `e6aa18a` holds the same tree. Its `src/`,
  `ac14998`, is that of `b71572e`, M4's harness commit; its `scorecard/`, `12a869c`, adds to M4's
  harness the reader's fixes (M4's deviations, above), the third run's seeds
  (`Declaration.third_confirmatory`, ADR 0101) and M5's pre-registration
  (`git log b71572e..e6aa18a -- src scorecard` lists them).
  The run locked causal-hybrid-control 1.3.1 and jax 0.12.0.
- **M3**, the evidence for the geo verdicts and the tuned rivals. Its pre-registration is
  `scorecard/preregistration/m3.md`, with `m3.json` and `m3_sizing.py` beside it. It was frozen on
  2026-11-17, before any confirmatory world was fitted, in the same then-private repository. Its
  SHA-256 is `d684b25ccae0e3574fe2f962097e399c9f400f0641e52431cf794e131763d924`, which
  `sha256sum scorecard/preregistration/m3.md` prints.
- **M3 has no public time-stamp before the run.** The repository was private when the file was
  frozen and while the run ran (2026-11-17 to 2026-11-20). Nothing public shows that the file was
  not changed after the results were known; the hash shows only that this file is the one the
  run declared.
- This repository starts from a fresh history at 0.1.0. The engine and the scorecard's harness in
  its first commit are the code M3 ran: their git tree hashes,
  `782ca30c324a5720c37c4b4e6856862d4f22a2f5` for `src/` and
  `32c5f8714d670141efc9f9c636fce410975ad2bb` for `scorecard/`, are those of the commit M3
  declared (`git rev-parse v0.1.0:src v0.1.0:scorecard`). M3 locked causal-hybrid-control 1.3.0;
  0.1.0 locks 1.3.1, whose changes the engine does not call.
- The bench is public: [causaldyn/causaldyn-bench](https://github.com/causaldyn/causaldyn-bench).
  M5 ran its commit `bc52170ac9c4069d9f843c0d7c45584c6320bcc3` (branch `m5`), M4 its commit
  `5ddfe0cf05170be034b50e0ef2d6d38daf2f0aa0` (branch `m4`), M3 its commit
  `67f92b7478b0a1423538d77e62d8198923c52a85` (branch `m3`), each rival's environment pinned by its
  lock file under `scripts/`.
- The harness is `scorecard/`. `python -m scorecard.run` declares a run, exports its worlds, runs
  every arm and scores every record (its docstring gives the environment); `python -m
  scorecard.claim read --recorded <roots>` reads the claim, the verdicts, Holm's losses and the
  recorded arms less every rival. M5 took 250 CPU-hours, M4 took 1 502 CPU-hours and
  M3 took 5 201 CPU-hours (Compute, below). Their per-world records are not published yet, so the
  readings cannot be re-run without re-running the worlds.
- **The tables of the claim, the worlds per cell, the words, the losses, the tuned rivals, the
  versions and the compute are written, not typed.** `scripts/readme_record.py extract` read them
  from each run's records and reading into `evidence/results/m5_record.json`,
  `evidence/results/m4_record.json` and `evidence/results/m3_record.json`, with the SHA-256 of every
  file they came from; `python scripts/readme_record.py` writes them here from those files, and
  `tests/test_readme.py` fails where a table differs from what it writes.

## Versions of every arm

What each arm's records state: the packages and the Python each record names, the commits of the
engine and the bench where the record names them, and the settings each record holds. A setting
marked "by family" took each value on some families: a history's length sets a kernel's or a
lag's weeks. The rest of each arm's setting is in its run's pre-registration's Arms table.

### M4

The records of `anchored floored` carry `anchored held`'s route setting, but the variant plays no
route (ADR 0099).

<!-- record:m4-versions -->
| Arm | Role | Version, as each record states it | Run with | How it was configured |
|---|---|---|---|---|
| `margin held` | the claim's arm at k = 0; families 0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 17, 18, 19, 20 | bench at `5ddfe0c`, mixdyn at `cc83d2a` | Python 3.14.10; causal-hybrid-control 1.3.1, jax 0.12.0, numpy 2.5.4, scipy 1.18.1 | planner water-filling; retentions 0.1, 0.3, 0.5, 0.7, 0.9; kernel weeks 6, 9, 156 or 1 (by family); AR(1) grid 0.3, 0.6, 0.8, 0.9, 0.95, 0.98; time guard 0.8 |
| `anchored floored` | the claim's arm at the tested rung, not routed; families 0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 17, 18, 19, 20 | bench at `5ddfe0c`, mixdyn at `cc83d2a` | Python 3.14.10; causal-hybrid-control 1.3.1, jax 0.12.0, numpy 2.5.4, scipy 1.18.1 | planner water-filling on a table; curves michaelis-menten, logistic; lift rows read 12, 3, 6 or 8 (by family); floor, a share of the least weekly spend 0.9; time guard 0.8 |
| `chc`, `ridge` | the engine's other arms, recorded beside the claim; families 0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 17, 18, 19, 20 | bench at `5ddfe0c`, mixdyn at `cc83d2a` | Python 3.14.10; causal-hybrid-control 1.3.1, jax 0.12.0, numpy 2.5.4, scipy 1.18.1 | each record's own settings |
| Meridian | rival; families 0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 17, 18, 20; 564 failure records, family 19 | Meridian 2.1.0 | Python 3.12.16; arviz 0.19.0, jax 0.11.2, jaxlib 0.11.2, numpy 2.3.5, pandas 2.3.3, tensorflow 2.21.0, tfp-nightly 0.26.0.dev20260130, xarray 2026.9.0 | sampling n_adapt 1000, n_burnin 500, n_chains 10, n_keep 1000; automatic knot selection on; ROI prior at k = 0: loc 0.2; scale 0.9; max lag 5, 8, 155 or 0 (by family); dtype float64 |
| PyMC-Marketing | rival; families 0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 17, 18, 19, 20 | PyMC-Marketing 1.3.0 | Python 3.12.16; arviz 1.3.0, numpy 2.4.6, pandas 3.1.0, pymc 6.3.2, pytensor 3.3.6, scipy 1.18.1 | model static; chains 4; draws 1000; adstock weeks 6, 9, 156 or 1 (by family); yearly harmonics 3 |
| Orbit | rival; families 0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 17, 18, 19, 20; 3 failure records, family 18 | Orbit 1.1.5.1 | Python 3.12.16; cmdstanpy 1.3.0, numpy 2.5.4, pandas 3.0.6, pyro-ppl 1.9.2, scipy 1.18.1, torch 2.14.1+cpu | model log-log KTR (Ng, Wang and Dai 2021); samples 400; steps KTR's default; holdout weeks 13; seasonality 52 |
<!-- /record:m4-versions -->

### M5

M5 fitted every rival on every world of its own: it reused no record of another run.

<!-- record:m5-versions -->
| Arm | Role | Version, as each record states it | Run with | How it was configured |
|---|---|---|---|---|
| `margin held` | the claim's arm at k = 0; families 1, 6, 14, 15 | bench at `bc52170`, mixdyn at `b864f6d` | Python 3.14.10; causal-hybrid-control 1.3.1, jax 0.12.0, numpy 2.5.4, scipy 1.18.1 | planner water-filling; retentions 0.1, 0.3, 0.5, 0.7, 0.9; kernel weeks 6 or 156 (by family); AR(1) grid 0.3, 0.6, 0.8, 0.9, 0.95, 0.98; time guard 0.8 |
| `anchored floored` | the claim's arm at the tested rung, not routed; family 15 | bench at `bc52170`, mixdyn at `b864f6d` | Python 3.14.10; causal-hybrid-control 1.3.1, jax 0.12.0, numpy 2.5.4, scipy 1.18.1 | planner water-filling on a table; curves michaelis-menten, logistic; lift rows read 12; floor, a share of the least weekly spend 0.9; time guard 0.8 |
| Meridian | rival; families 1, 6, 14, 15 | Meridian 2.1.0 | Python 3.12.16; arviz 0.19.0, jax 0.11.2, jaxlib 0.11.2, numpy 2.3.5, pandas 2.3.3, tensorflow 2.21.0, tfp-nightly 0.26.0.dev20260130, xarray 2026.9.0 | sampling n_adapt 1000, n_burnin 500, n_chains 10, n_keep 1000; automatic knot selection on; ROI prior at k = 0: loc 0.2; scale 0.9; max lag 5 or 155 (by family); dtype float64 |
| PyMC-Marketing | rival; families 1, 6, 14, 15; 22 failure records, family 15 | PyMC-Marketing 1.3.0 | Python 3.12.16; arviz 1.3.0, numpy 2.4.6, pandas 3.1.0, pymc 6.3.2, pytensor 3.3.6, scipy 1.18.1 | model static; chains 4; draws 1000; adstock weeks 6 or 156 (by family); yearly harmonics 3 |
| Orbit | rival; families 1, 6, 14, 15 | Orbit 1.1.5.1 | Python 3.12.16; cmdstanpy 1.3.0, numpy 2.5.4, pandas 3.0.6, pyro-ppl 1.9.2, scipy 1.18.1, torch 2.14.1+cpu | model log-log KTR (Ng, Wang and Dai 2021); samples 400; steps KTR's default; holdout weeks 13; seasonality 52 |
<!-- /record:m5-versions -->

### M3: the geo verdicts and the tuned rivals

The rivals' records M3 reused from M2 name the same versions.

<!-- record:versions -->
| Arm | Role | Version, as each record states it | Run with | How it was configured |
|---|---|---|---|---|
| `margin held` | the claim's arm at k = 0; families 0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15, 17, 18, 19, 20 | bench at `67f92b7`, mixdyn at `183ca75` | Python 3.14.10; causal-hybrid-control 1.3.0, jax 0.12.0, numpy 2.5.4, scipy 1.18.1 | planner water-filling; retentions 0.1, 0.3, 0.5, 0.7, 0.9; kernel weeks 6, 9, 156 or 1 (by family); AR(1) grid 0.3, 0.6, 0.8, 0.9, 0.95, 0.98; time guard 0.8 |
| `anchored held` | the claim's arm at the tested rung, `gs held`: its router; families 0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15, 17, 18, 19, 20 | bench at `67f92b7`, mixdyn at `183ca75` | Python 3.14.10; causal-hybrid-control 1.3.0, jax 0.12.0, numpy 2.5.4, scipy 1.18.1 | planner water-filling on a table; curves michaelis-menten, logistic; lift rows read 12, 3, 6 or 8 (by family); routes a flagged world to stacked; time guard 0.8 |
| `stacked` | `gs held`'s routed plan, the Bayesian engine's; families 0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 20 | bench at `67f92b7`, mixdyn at `183ca75` | Python 3.14.10; causal-hybrid-control 1.3.0, jax 0.12.0, numpy 2.5.4, scipy 1.18.1 | sampler chains 4, draws 1000, warmup 1000; curves hill, tanh; planner plan_posterior |
| `geo eb` | the claim's arm on the geo families; families 8, 9, 10 | bench at `67f92b7`, mixdyn at `183ca75` | Python 3.14.10; causal-hybrid-control 1.3.0, jax 0.12.0, numpy 2.5.4, scipy 1.18.1 | planner plan_mixture; retentions 0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9; curvatures 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, inf; sweeps 3 |
| `chc`, `decisions`, `dlm`, `geo-dlm`, `read-sc`, `read-sdid`, `read-tbr`, `ridge`, `switchback`, `voi` | the engine's other arms, recorded beside the claim; families 0, 2, 3, 5, 7, 8, 9, 10, 12, 13, 17, 18, 19, 20 | bench at `67f92b7`, mixdyn at `183ca75` | Python 3.14.10; causal-hybrid-control 1.3.0, jax 0.12.0, numpy 2.5.4, scipy 1.18.1 | each record's own settings |
| Meridian | rival; families 0, 2, 3, 5, 7, 8, 9, 10, 12, 13, 17, 18, 20; M2's records reused on 1, 4, 6, 14, 15; 360 failure records, family 19 | Meridian 2.1.0 | Python 3.12.16; arviz 0.19.0, jax 0.11.2, jaxlib 0.11.2, numpy 2.3.5, pandas 2.3.3, tensorflow 2.21.0, tfp-nightly 0.26.0.dev20260130, xarray 2026.9.0 | sampling n_adapt 1000, n_burnin 500, n_chains 10, n_keep 1000; automatic knot selection on; ROI prior at k = 0: loc 0.2; scale 0.9; max lag 5, 7, 8 or 0 (by family); dtype float64 |
| Meridian tuned | tuned rival; families 0, 2, 3, 5, 7, 8, 9, 10, 12, 13, 17, 18, 20; 360 failure records, family 19 | Meridian 2.1.0 | Python 3.12.16; arviz 0.19.0, jax 0.11.2, jaxlib 0.11.2, numpy 2.3.5, pandas 2.3.3, tensorflow 2.21.0, tfp-nightly 0.26.0.dev20260130, xarray 2026.9.0 | sampling n_adapt 1000, n_burnin 500, n_chains 10, n_keep 1000; automatic knot selection off; media prior contribution; knots, weeks apart at most 13; max lag 5, 7, 8 or 0 (by family); dtype float64 |
| PyMC-Marketing | rival; families 0, 2, 3, 5, 7, 8, 9, 10, 12, 13, 17, 18, 19, 20; M2's records reused on 1, 4, 6, 14, 15; 49 failure records, families 8, 9, 10, 15 | PyMC-Marketing 1.3.0 | Python 3.12.16; arviz 1.3.0, numpy 2.4.6, pandas 3.1.0, pymc 6.3.2, pytensor 3.3.6, scipy 1.18.1 | model static; chains 4; draws 1000; adstock weeks 6, 8, 9 or 1 (by family); yearly harmonics 3 |
| Orbit | rival; families 0, 2, 3, 5, 7, 8, 9, 10, 12, 13, 18, 19, 20; M2's records reused on 1, 4, 6, 14, 15; 242 failure records, families 17, 18 | Orbit 1.1.5.1 | Python 3.12.16; cmdstanpy 1.3.0, numpy 2.5.4, pandas 3.0.6, pyro-ppl 1.9.2, scipy 1.18.1, torch 2.14.1+cpu | model log-log KTR (Ng, Wang and Dai 2021); samples 400; steps KTR's default; holdout weeks 13; seasonality 52 |
| Robyn | rival, read beside the claim; families 9, 12, 13 | Robyn 3.12.1 | R version 4.6.2; StanHeaders 2.39.1, doParallel 1.0.17, doRNG 1.8.6.3, dplyr 1.2.1, foreach 1.5.2, glmnet 5.1, jsonlite 2.0.0, nloptr 2.2.1, prophet 1.1.7, reticulate 1.47.0, rstan 2.32.7; Python 3.13.18, nevergrad 1.0.12, numpy 2.5.3 | ranges genre; iterations 2000; trials 5 or 10 (by family); cores 4; date range last_13; calibration total |
| Robyn tuned | tuned rival, read beside the claim; families 9, 12, 13 | Robyn 3.12.1 | R version 4.6.2; StanHeaders 2.39.1, doParallel 1.0.17, doRNG 1.8.6.3, dplyr 1.2.1, foreach 1.5.2, glmnet 5.1, jsonlite 2.0.0, nloptr 2.2.1, prophet 1.1.7, reticulate 1.47.0, rstan 2.32.7; Python 3.13.18, nevergrad 1.0.12, numpy 2.5.3 | ranges genre; iterations 2000; trials 5 or 10 (by family); cores 4; date range last_13; calibration total; ts_validation on |
<!-- /record:versions -->

## Compute

Each process a run started is one line of its root's `logs/costs.jsonl`: one arm on one world, at
the rungs it planned. Its core-hours are its scope's CPU time, its peak memory its scope's
`memory.peak` (the memory charged to the scope, page cache included: not the same measure as a peak
RSS), and its wall the time from its start to its end. A failed attempt that ended by itself is a
line too; a process stopped at its timeout or by one of the box's spot interruptions wrote none, so
its CPU time is not counted here. The CPU model and the threads (`OMP_NUM_THREADS`) are what each
record's `cost` names; the CPU quota is the CPUs the process was given, its cost class's in the
stage's plan. Each run's processes shared one box, M5's and M4's a c6a.48xlarge and M3's an
m6a.48xlarge (each pre-registration's Compute): the figures say what was spent under that load, not
how fast a tool is.

### M4

<!-- record:m4-compute -->
One box of 192 CPUs and 370 GiB ran every process, admitting up to 188 CPUs of quota at a time. It
was launched 3 times: the stage resumed after each of 2 spot interruptions.

| Family | CPU model | Threads | Processes | Peak memory a process, median / most | Wall a process, median / most | Core-hours: engine | rivals | all |
|---|---|---|---|---|---|---|---|---|
| 0 Track M v2 | AMD EPYC 7R13 Processor | 1 | 3 100 | 0.73 / 9.49 GiB | 89 s / 380 s | 11.6 | 80.4 | 91.9 |
| 1 Drift | AMD EPYC 7R13 Processor | 1 | 5 220 | 0.57 / 8.03 GiB | 88 s / 339 s | 19.9 | 135.5 | 155.3 |
| 2 Spend confounded with demand | AMD EPYC 7R13 Processor | 1 | 2 699 | 0.72 / 10.02 GiB | 91 s / 389 s | 10.5 | 72.1 | 82.5 |
| 3 Collinear channels | AMD EPYC 7R13 Processor | 1 | 2 700 | 0.66 / 8.32 GiB | 93 s / 337 s | 10.6 | 71.1 | 81.7 |
| 4 S-curves | AMD EPYC 7R13 Processor | 1 | 5 220 | 0.41 / 10.18 GiB | 78 s / 435 s | 18.1 | 124.2 | 142.3 |
| 5 Budget follows performance | AMD EPYC 7R13 Processor | 1 | 2 400 | 0.57 / 7.39 GiB | 91 s / 346 s | 9.2 | 62.7 | 72.0 |
| 6 Truth outside the observed range | AMD EPYC 7R13 Processor | 1 | 2 400 | 0.57 / 10.20 GiB | 93 s / 429 s | 9.4 | 71.6 | 81.0 |
| 7 Short series | AMD EPYC 7R13 Processor | 1 | 2 519 | 0.42 / 7.06 GiB | 77 s / 285 s | 8.3 | 56.5 | 64.9 |
| 12 Meridian's home ground | AMD EPYC 7R13 Processor | 1 | 4 600 | 0.58 / 10.39 GiB | 66 s / 443 s | 13.6 | 92.0 | 105.6 |
| 13 PyMC-Marketing's home ground | AMD EPYC 7R13 Processor | 1 | 5 200 | 0.43 / 7.29 GiB | 65 s / 335 s | 15.1 | 101.5 | 116.6 |
| 14 Robyn's home ground | AMD EPYC 7R13 Processor | 1 | 5 200 | 0.64 / 12.30 GiB | 70 s / 19 min | 15.6 | 220.2 | 235.8 |
| 17 Twins | AMD EPYC 7R13 Processor | 1 | 2 399 | 0.45 / 1.42 GiB | 73 s / 194 s | 6.3 | 48.3 | 54.6 |
| 18 Long term | AMD EPYC 7R13 Processor | 1 | 2 401 | 0.45 / 10.23 GiB | 94 s / 405 s | 9.2 | 72.2 | 81.4 |
| 19 Flat | AMD EPYC 7R13 Processor | 1 | 2 820 | 0.48 / 0.88 GiB | 49 s / 223 s | 10.6 | 51.2 | 61.9 |
| 20 Promotions | AMD EPYC 7R13 Processor | 1 | 2 400 | 0.59 / 7.28 GiB | 94 s / 347 s | 9.6 | 64.7 | 74.3 |
| all | | | 51 278 | | | 177.4 | 1 324.4 | 1 501.8 |

| Arm | Role | CPU quota | Rungs a process | Families | Processes | Peak memory a process, median / most | Wall a process, median / most | Core-hours |
|---|---|---|---|---|---|---|---|---|
| `meridian` | rival | 1.2 | 1 | 0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 17, 18, 19, 20 | 10 254 | 2.38 / 12.30 GiB | 178 s / 19 min | 648.0 |
| `pymc-marketing` | rival | 1.5 | 1 | 0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 17, 18, 19, 20 | 10 255 | 0.75 / 1.11 GiB | 106 s / 230 s | 456.9 |
| `orbit` | rival | 1 | 1 | 0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 17, 18, 19, 20 | 10 258 | 0.41 / 0.82 GiB | 81 s / 123 s | 219.5 |
| `chc` | engine | 1 | 2 | 0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 17, 18, 19, 20 | 5 128 | 0.88 / 0.94 GiB | 76 s / 154 s | 105.9 |
| `ridge` | engine | 1 | 2 | 0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 17, 18, 19, 20 | 5 128 | 0.25 / 0.26 GiB | 29 s / 95 s | 43.2 |
| `margin` | engine | 1 | 2 | 0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 17, 18, 19, 20 | 5 127 | 0.17 / 0.18 GiB | 11 s / 17 s | 15.0 |
| `anchored` | engine | 1 | 1 | 0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 17, 18, 19, 20 | 5 128 | 0.17 / 0.18 GiB | 10 s / 15 s | 13.3 |
<!-- /record:m4-compute -->

### M5

<!-- record:m5-compute -->
One box of 192 CPUs and 370 GiB ran every process, admitting up to 188 CPUs of quota at a time. It
was launched once, and no spot interruption stopped it.

| Family | CPU model | Threads | Processes | Peak memory a process, median / most | Wall a process, median / most | Core-hours: engine | rivals | all |
|---|---|---|---|---|---|---|---|---|
| 1 Drift | AMD EPYC 7R13 Processor | 1 | 960 | 0.57 / 9.50 GiB | 69 s / 326 s | 0.6 | 27.5 | 28.1 |
| 6 Truth outside the observed range | AMD EPYC 7R13 Processor | 1 | 960 | 0.63 / 8.40 GiB | 71 s / 323 s | 0.6 | 30.8 | 31.4 |
| 14 Robyn's home ground | AMD EPYC 7R13 Processor | 1 | 960 | 0.77 / 11.40 GiB | 100 s / 16 min | 0.6 | 59.7 | 60.3 |
| 15 siMMMulator | AMD EPYC 7R13 Processor | 1 | 1 912 | 0.64 / 6.23 GiB | 86 s / 22 min | 1.1 | 128.6 | 129.8 |
| all | | | 4 792 | | | 3.0 | 246.6 | 249.6 |

| Arm | Role | CPU quota | Rungs a process | Families | Processes | Peak memory a process, median / most | Wall a process, median / most | Core-hours |
|---|---|---|---|---|---|---|---|---|
| `meridian` | rival | 1.2 | 1 | 1, 6, 14, 15 | 1 200 | 3.38 / 11.40 GiB | 371 s / 16 min | 130.9 |
| `pymc-marketing` | rival | 1.5 | 1 | 1, 6, 14, 15 | 1 192 | 0.78 / 0.81 GiB | 141 s / 22 min | 96.2 |
| `orbit` | rival | 1 | 1 | 1, 6, 14, 15 | 1 200 | 0.64 / 0.73 GiB | 60 s / 89 s | 19.6 |
| `margin` | engine | 1 | 1 or 2 | 1, 6, 14, 15 | 960 | 0.17 / 0.18 GiB | 9 s / 13 s | 2.4 |
| `anchored` | engine | 1 | 1 | 15 | 240 | 0.17 / 0.17 GiB | 8 s / 10 s | 0.5 |
<!-- /record:m5-compute -->

### M3

<!-- record:compute -->
One box of 192 CPUs and 740 GiB ran every process, admitting up to 188 CPUs of quota at a time. It
was launched 4 times: the stage resumed after each of 3 spot interruptions.

| Family | CPU model | Threads | Processes | Peak memory a process, median / most | Wall a process, median / most | Core-hours: engine | rivals | all |
|---|---|---|---|---|---|---|---|---|
| 0 Track M v2 | AMD EPYC 7R13 Processor | 1 | 5 489 | 0.87 / 9.15 GiB | 138 s / 33 min | 315.8 | 139.2 | 455.0 |
| 1 Drift | AMD EPYC 7R13 Processor | 1 | 720 | 0.17 / 0.89 GiB | 14 s / 10 min | 35.4 | 0.0 | 35.4 |
| 2 Spend confounded with demand | AMD EPYC 7R13 Processor | 1 | 3 600 | 0.85 / 10.67 GiB | 141 s / 39 min | 196.2 | 100.9 | 297.1 |
| 3 Collinear channels | AMD EPYC 7R13 Processor | 1 | 4 050 | 0.85 / 8.61 GiB | 140 s / 48 min | 229.5 | 111.2 | 340.7 |
| 4 S-curves | AMD EPYC 7R13 Processor | 1 | 720 | 0.17 / 0.89 GiB | 14 s / 12 min | 33.9 | 0.0 | 33.9 |
| 5 Budget follows performance | AMD EPYC 7R13 Processor | 1 | 3 600 | 0.85 / 7.66 GiB | 141 s / 37 min | 193.2 | 99.1 | 292.3 |
| 6 Truth outside the observed range | AMD EPYC 7R13 Processor | 1 | 720 | 0.17 / 0.91 GiB | 14 s / 11 min | 34.3 | 0.0 | 34.3 |
| 7 Short series | AMD EPYC 7R13 Processor | 1 | 3 600 | 0.71 / 9.67 GiB | 129 s / 25 min | 129.5 | 86.0 | 215.4 |
| 8 Geo | AMD EPYC 7R13 Processor | 1 | 3 230 | 0.69 / 20.23 GiB | 333 s / 42 min | 174.2 | 456.6 | 630.7 |
| 9 Spillover between geos | AMD EPYC 7R13 Processor | 1 | 3 082 | 1.23 / 20.49 GiB | 324 s / 72 min | 10.5 | 880.8 | 891.4 |
| 10 Few geos | AMD EPYC 7R13 Processor | 1 | 2 880 | 0.74 / 19.78 GiB | 130 s / 49 min | 9.5 | 296.9 | 306.4 |
| 12 Meridian's home ground | AMD EPYC 7R13 Processor | 1 | 3 701 | 0.86 / 11.60 GiB | 132 s / 72 min | 176.7 | 286.6 | 463.3 |
| 13 PyMC-Marketing's home ground | AMD EPYC 7R13 Processor | 1 | 3 706 | 0.87 / 9.30 GiB | 131 s / 72 min | 175.3 | 291.2 | 466.5 |
| 14 Robyn's home ground | AMD EPYC 7R13 Processor | 1 | 720 | 0.17 / 0.89 GiB | 13 s / 10 min | 32.8 | 0.0 | 32.8 |
| 15 siMMMulator 1.1.0 | AMD EPYC 7R13 Processor | 1 | 300 | 0.17 / 0.95 GiB | 13 s / 24 min | 14.9 | 0.0 | 14.9 |
| 17 Twins | AMD EPYC 7R13 Processor | 1 | 1 800 | 0.72 / 2.29 GiB | 112 s / 16 min | 62.8 | 39.0 | 101.8 |
| 18 Long term | AMD EPYC 7R13 Processor | 1 | 1 800 | 0.88 / 9.60 GiB | 159 s / 40 min | 108.2 | 54.8 | 163.0 |
| 19 Flat | AMD EPYC 7R13 Processor | 1 | 2 700 | 0.52 / 5.86 GiB | 88 s / 33 min | 148.9 | 39.6 | 188.4 |
| 20 Promotions | AMD EPYC 7R13 Processor | 1 | 2 700 | 0.85 / 6.66 GiB | 150 s / 39 min | 159.1 | 78.9 | 238.0 |
| all | | | 49 118 | | | 2 240.6 | 2 960.8 | 5 201.3 |

| Arm | Role | CPU quota | Rungs a process | Families | Processes | Peak memory a process, median / most | Wall a process, median / most | Core-hours |
|---|---|---|---|---|---|---|---|---|
| `dlm` | engine | 1 | 1 | 0, 2, 3, 5, 7, 12, 13, 17, 18, 19, 20 | 4 839 | 3.55 / 6.80 GiB | 18 min / 48 min | 1 424.8 |
| `meridian` | rival | 1.2 | 1 | 0, 2, 3, 5, 7, 8, 9, 10, 12, 13, 17, 18, 19, 20 | 6 279 | 3.09 / 20.49 GiB | 218 s / 46 min | 786.6 |
| `meridian tuned` | rival | 1.2 | 1 | 0, 2, 3, 5, 7, 8, 9, 10, 12, 13, 17, 18, 19, 20 | 6 278 | 2.05 / 20.47 GiB | 182 s / 43 min | 692.4 |
| `pymc-marketing` | rival | 1.5 | 1 | 0, 2, 3, 5, 7, 8, 9, 10, 12, 13, 17, 18, 19, 20 | 6 259 | 0.75 / 1.25 GiB | 139 s / 49 min | 519.4 |
| `bayes` | engine | 1 | 1 | 0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 20 | 3 600 | 0.85 / 0.95 GiB | 450 s / 38 min | 502.1 |
| `robyn` | rival | 3 | 1 | 9, 12, 13 | 172 | 1.32 / 2.03 GiB | 39 min / 72 min | 401.9 |
| `robyn tuned` | rival | 3 | 1 | 9, 12, 13 | 169 | 1.37 / 2.02 GiB | 30 min / 72 min | 401.4 |
| `orbit` | rival | 1 | 1 | 0, 2, 3, 5, 7, 8, 9, 10, 12, 13, 17, 18, 19, 20 | 6 282 | 0.41 / 0.83 GiB | 95 s / 143 s | 159.2 |
| `chc` | engine | 1 | 2 | 0, 2, 3, 5, 7, 8, 9, 10, 12, 13, 17, 18, 19, 20 | 3 220 | 0.88 / 0.94 GiB | 89 s / 153 s | 78.2 |
| `switchback` | engine | 1 | 1 | 8 | 60 | 0.72 / 0.76 GiB | 36 min / 42 min | 35.5 |
| `ridge` | engine | 1 | 2 | 0, 2, 3, 5, 7, 8, 9, 10, 12, 13, 17, 18, 19, 20 | 3 140 | 0.25 / 0.26 GiB | 36 s / 118 s | 32.7 |
| `read-sdid` | engine | 1 | 1 | 8 | 60 | 0.69 / 0.73 GiB | 33 min / 37 min | 31.7 |
| `read-sc` | engine | 1 | 1 | 8 | 60 | 0.69 / 0.73 GiB | 32 min / 37 min | 31.6 |
| `read-tbr` | engine | 1 | 1 | 8 | 60 | 0.69 / 0.73 GiB | 32 min / 37 min | 31.6 |
| `voi` | engine | 1 | 1 | 0 | 120 | 1.02 / 1.11 GiB | 14 min / 17 min | 30.2 |
| `decisions` | engine | 1 | 1 | 0 | 60 | 0.88 / 0.94 GiB | 12 min / 14 min | 12.5 |
| `margin` | engine | 1 | 1 or 2 | 0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15, 17, 18, 19, 20 | 3 480 | 0.17 / 0.18 GiB | 13 s / 17 s | 12.4 |
| `anchored` | engine | 1 | 1 | 0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15, 17, 18, 19, 20 | 3 480 | 0.17 / 0.18 GiB | 12 s / 15 s | 10.9 |
| `geo-eb` | engine | 1 | 1 | 8, 9, 10 | 1 440 | 0.17 / 0.18 GiB | 11 s / 15 s | 4.4 |
| `geo-dlm` | engine | 1 | 1 | 8 | 60 | 0.30 / 0.30 GiB | 128 s / 170 s | 2.1 |

Families 1, 4, 6, 14, 15 read the rivals' records of M2, whose compute is not counted here; those
records name AMD EPYC 7R13 Processor. The engine's core-hours include the arms recorded beside the
claim, read by no comparison: 50.9 on family 0, 164.7 on family 8.
<!-- /record:compute -->

## Real-data validation: none, and why

No public panel carries the truth an engine would be scored against: a real market's returns are
never observed, only estimated. Every claim about the engine is therefore measured on simulated
worlds whose truth is known, the bench's families, and no claim here is about real data.

What real data can score is a prediction of a lift or geo test the model did not read: a test's
reading is a noisy but unbiased measurement of a quantity every engine predicts.
`scripts/real_data_protocol.py` is the protocol a partner would run on one weekly panel with at
least two such tests. Each test is withheld in turn; every engine fits the panel and the other
tests and predicts the withheld test's lift; the cells are `|predicted - test| / SE`, each test's
standard error, and the share of tests whose 95 % interval holds the prediction. It runs end to
end on a synthetic stand-in (`python scripts/real_data_protocol.py`), and
`tests/test_real_data_protocol.py` runs it on a small one. No partner has run it yet, so no real
panel's cells exist.
