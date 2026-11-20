# 0056. A switchback test read by its own randomisation

Status: accepted, 2026-10-29; adds to the geo tests of ADRs 0047 to 0049 a test randomised in time,
for a channel no geography can cut, read into the same `LiftEstimate`; amended 2026-10-31: "hac"
reads a lag-0 bandwidth beside Newey and West's, its over-rejection is the kernel's, "hac" is a
reporting fallback without a size, and the lift counts the carry-out past the test; amended
2026-11-06: the audit's cells, reader by reader, under Consequences; and an opt-in `burn_in`, the
half-panel jackknife's error as `method="jackknife"`, and `carryover_order`, the two-epoch test of
the carryover's order by the design's own randomisation; the jackknife's bias correction is not
adopted

## Context

A geo test (ADRs 0047, 0049) turns a channel off in some geos and reads the others as their
counterfactual. A channel bought nationally, national TV say, or a brand sold in one market, has no
geo to leave out. What can still be randomised is time: the channel is switched on and off over
blocks of periods, and the periods off are the periods on's counterfactual (Bojinov, Simchi-Levi
and Zhao 2023, "Design and analysis of switchback experiments", Management Science 69). Three
choices decide whether such a test reads the channel's lift:

- **Carryover.** A channel's effect outlasts its spend: under a geometric adstock of retention
  `lam` the first periods after a switch still read the previous block's state (Hu and Wager
  2022, "Switchback experiments under geometric mixing"). The plain difference of the periods on
  and off is then biased towards 0. Over every arrangement of 6 of 12 blocks of 4 periods on, the
  channel off before the test, its bias is a share
  `-(G / L) (1 + 1/B - r (1 - r^(B-1)) / (B (B-1) (1 - r)))` of the effect a period, `G = lam (1 -
  lam^L) / (1 - lam)`, `r = lam^L`: -0.2538 at `lam = 0.5` and -0.6365 at `lam = 0.8`
  (validation/switchback_carryover.mac, which checks it against every arrangement of 4 to 6
  blocks and of the 12 here; the tests hold 2000 simulated tests' mean to it within 3 Monte Carlo
  standard errors).
- **The noise's dependence over time.** A national KPI's noise carries over from period to period
  and so from block to block. An error read from few blocks as if they were independent can be
  too small where it persists across them, and one read from a kernel over few blocks is too small
  even where it does not (amended 2026-10-31; see the evidence).
- **The sample.** A test of 48 weeks in blocks of 4 has 12 blocks; any error that rests on many
  blocks rests on asymptotics it does not reach.

## Decision

- **`switchback_design(periods, block, washout, *, seed, share=0.5)`** cuts the periods, dates,
  into blocks of `block` and turns the channel on in exactly `round_half_up(share * blocks)`, in an
  order drawn from `numpy.random.SeedSequence(seed).spawn(1)[0]`, the child `assign` draws from.
  Exactly that many on (a balanced design), not each block on with probability `share`, so that no
  draw leaves a state with one block. It refuses a washout at or above the block, fewer than 4
  blocks, periods that do not cut into whole blocks and a share outside `(0, 1)` or one that turns
  no block on or all of them. `SwitchbackDesign` is frozen, its dates `datetime64[D]` as
  `GeoTest`'s are.
- **The washout.** The first `washout` periods of every block after a switch are left out of the
  reading; a block in the same state as the one before it keeps all its periods, its carryover
  being its own state's. The first block is read as after a switch: the state before the test is
  not the design's. The lift still counts every period on: it is the mean KPI over the kept
  periods on less the mean over the kept periods off, times the periods on, so the KPI the
  periods on gained, the carry-out of the spend included, as a `LiftEstimate` over the test's
  periods against a `planned` spend that holds every period on at the periods off's mean. The
  lift is the steady state's (amended 2026-10-31): the effect a period on times the periods on,
  the carry-out past the test's last period included, which the `LiftEstimate`'s readout, the
  test's periods, does not hold. Noise-free, at an effect of 1 a period over 12 blocks of 4 and
  the designs of seeds 0 to 1999, the lift inside the test's periods is 23.49 at retention 0.5
  and 21.97 at 0.8 on the mean (23.03 and 20.89 where the last block is on), so the reading's 24
  is high by 2.1 % and 8.4 % against the readout. The estimate is not moved; a readout padded
  with periods off for the kernel's length, as the geo readers' cooldown is, would close the gap.
- **`read_switchback(kpi, spend, design, *, level=0.95, method="randomization",
  bandwidth="newey-west")`** sums a KPI by
  geo as the geo readers sum it and reads it by one of two methods:
  - **"randomization"**, the default: the difference over every arrangement of the design's
    blocks with as many on, each read with the periods its own switches keep, enumerated up to
    100 000 and otherwise the observed one and 10 000 drawn from `SeedSequence(seed).spawn(2)[1]`.
    The p-value is the share at least as far from 0 as the observed difference; the interval
    inverts the same test under a constant effect `tau` a period on. Every arrangement's
    difference is linear in `tau`, so each fails the test on one interval of `tau` and the
    interval's ends are found exactly: the first `tau` on each side of the estimate where the
    test rejects. The KPI is held fixed and only the assignment is random, so the size holds
    whatever the noise.
  - **"hac"**: the blocks' means over their kept periods, the difference a weighted sum of them,
    its error read as a Student t at `B - 2` degrees of freedom, by one of two bandwidths.
    `bandwidth="newey-west"`, the default: Newey and West's (1987, Econometrica 55) over the
    blocks' scores, Bartlett weights to lag `floor(4 (B / 100)^(2/9))` (Newey and West 1994,
    Review of Economic Studies 61), times `B / (B - 2)`. `bandwidth="lag-0"` (amended
    2026-10-31): no lag, the pooled two-sample error of the block means `m_b`, each weighted by
    its kept periods `n_b`, `sum_b n_b (m_b - m_state)^2 / (B - 2) * (1 / N_on + 1 / N_off)`,
    `N` a state's kept periods, exact under a normal noise independent from period to period.
    Neither has a size on every cell of the evidence: "hac" is a reporting fallback, and the
    method that tests is "randomization".
- **The estimate's error** is widened so that its normal 95 % interval holds the reading's 95 %
  interval, as ADR 0047's are: `se t_{B-2}(0.975) / z(0.975)` for "hac", and for
  "randomization", whose interval need not be symmetric about the lift, the farther end's distance
  over `z(0.975)`. A design whose arrangements are too few for the test to reject at 95 %, 4 or 6
  blocks, is refused by "randomization"; "hac" reads it, without a size at either bandwidth.
- **A carryover check** is reported, not used: over the blocks after a switch that keep two
  periods or more, the first kept period less the rest's mean, the mean over blocks on less the
  mean over blocks off. Below 0 for a channel that raises the KPI, it says the washout is too
  short.

### The evidence

A national KPI of level 10, a stationary AR(1) noise of variance 1 at `rho`, and `effect` times
the channel's adstock at retention `lam`, normalised to read `effect` a period once the channel has
been on for good, the channel off before the test; 12 blocks of 4 periods (924 arrangements), or
8 (70). 4000 simulated tests a cell (seeds 0 to 3999, the design and the noise each drawn from the
seed, shared by every cell), a 95 % interval; the binomial standard error of a 5 % rate is 0.0034
and the band `0.05 +- 3 SE` is [0.0397, 0.0603].

The size at lift 0, the share of intervals that miss 0 (randomization / hac). The channel does
nothing, so its retention does not enter:

| blocks | `rho` | washout 0 | washout 1 | washout 2 |
| --- | --- | --- | --- | --- |
| 12 | 0 | 0.0520 / 0.0903 | 0.0545 / 0.0938 | 0.0520 / 0.1000 |
| 12 | 0.5 | 0.0553 / 0.0945 | 0.0533 / 0.1093 | 0.0513 / 0.1290 |
| 12 | 0.8 | 0.0563 / 0.0925 | 0.0533 / 0.1153 | 0.0510 / 0.1513 |
| 12 | 0.95 | 0.0510 / 0.0858 | 0.0535 / 0.1165 | 0.0513 / 0.1638 |
| 8 | 0 | 0.0278 / 0.1038 | 0.0300 / 0.1090 | |
| 8 | 0.5 | 0.0300 / 0.1060 | 0.0310 / 0.1205 | |
| 8 | 0.8 | 0.0278 / 0.1070 | 0.0280 / 0.1350 | |
| 8 | 0.95 | 0.0288 / 0.0950 | 0.0305 / 0.1303 | |

"randomization" passes every cell; it is exact by construction (over the 924 arrangements of one
fixed KPI it rejects 46, 0.0498), and the cells' common seeds move their Monte Carlo errors
together: the two cells the tests read, over 10 000 more seeds, rejected 0.0445 to 0.0565 in
batches of 2000. Over 8 blocks its 70 arrangements reject at most 3 of 70 of the time without
ties, and it is conservative. "hac" fails every cell, 0.086 to 0.164 over 12 blocks and 0.095 to
0.135 over 8, worst at a high `rho` with a washout. Its over-rejection at `rho = 0` is the
kernel's (amended 2026-10-31; the text read it as the noise persisting across blocks): the blocks'
scores are centred on their state's mean, so under a noise independent from period to period
their cross products are negative and every lag the Bartlett kernel reads takes from the
variance. As an exact quadratic form under such a noise, over the designs of seeds 0 to 199, the
expected Newey-West variance is 0.766, 0.686 and 0.839 of the difference's over 4, 6 and 12
blocks of 4 without a washout (lags 1, 1 and 2), and 0.824 over 12 with a washout of 1; at lag 0
it is 1.000 in each.

The two bandwidths at lift 0 (the "hac" rows above use the first), the share of 95 % intervals
that miss 0, newey-west / lag-0, blocks of 4, the same 4000 seeds a cell, band [0.0397, 0.0603]:

| blocks | `rho` | washout 0 | washout 1 |
| --- | --- | --- | --- |
| 4 | 0 | 0.0783 / 0.0480 | 0.0813 / 0.0505 |
| 4 | 0.5 | 0.0865 / 0.0535 | 0.0985 / 0.0563 |
| 4 | 0.8 | 0.0960 / 0.0515 | 0.1087 / 0.0592 |
| 6 | 0 | 0.1197 / 0.0447 | 0.1168 / 0.0440 |
| 6 | 0.5 | 0.1205 / 0.0512 | 0.1388 / 0.0542 |
| 6 | 0.8 | 0.1328 / 0.0595 | 0.1515 / 0.0717 |
| 12 | 0 | 0.0902 / 0.0508 | 0.0938 / 0.0558 |
| 12 | 0.5 | 0.0945 / 0.0537 | 0.1092 / 0.0703 |
| 12 | 0.8 | 0.0925 / 0.0568 | 0.1153 / 0.0767 |

"newey-west" fails all 18 cells. "lag-0" passes 15 and fails 3, each with a washout and a noise
that persists, `rho` 0.8 over 6 blocks and 0.5 and 0.8 over 12, where the noise does carry
across blocks. Neither bandwidth holds its size on every cell, so neither is made the test: the
default stays "randomization", "hac" stays a reporting fallback without a size, and its default
bandwidth stays "newey-west", whose readings are unchanged. Where a design is too small to
enumerate, "lag-0" is the closer of the two to its size.

With an effect of 1 a period, the share of intervals that miss the lift the channel made (the
effect times the periods on), randomization / hac, over 12 blocks:

| `rho` | `lam` | washout 0 | washout 1 | washout 2 |
| --- | --- | --- | --- | --- |
| 0 | 0 | 0.0520 / 0.0903 | 0.0545 / 0.0938 | 0.0520 / 0.1000 |
| 0 | 0.5 | 0.1245 / 0.1793 | 0.0688 / 0.1180 | 0.0528 / 0.1015 |
| 0 | 0.8 | 0.4525 / 0.5488 | 0.3033 / 0.4125 | 0.1778 / 0.2958 |
| 0.5 | 0 | 0.0553 / 0.0945 | 0.0533 / 0.1093 | 0.0513 / 0.1290 |
| 0.5 | 0.5 | 0.0960 / 0.1518 | 0.0628 / 0.1230 | 0.0518 / 0.1328 |
| 0.5 | 0.8 | 0.2830 / 0.3760 | 0.1900 / 0.2915 | 0.1253 / 0.2433 |
| 0.8 | 0 | 0.0563 / 0.0925 | 0.0533 / 0.1153 | 0.0510 / 0.1513 |
| 0.8 | 0.5 | 0.0905 / 0.1518 | 0.0565 / 0.1375 | 0.0520 / 0.1605 |
| 0.8 | 0.8 | 0.2578 / 0.3658 | 0.1813 / 0.2988 | 0.1223 / 0.2675 |
| 0.95 | 0 | 0.0510 / 0.0858 | 0.0535 / 0.1165 | 0.0513 / 0.1638 |
| 0.95 | 0.5 | 0.1173 / 0.1878 | 0.0713 / 0.1505 | 0.0565 / 0.1740 |
| 0.95 | 0.8 | 0.3768 / 0.5080 | 0.2625 / 0.4180 | 0.1685 / 0.3630 |

Without carryover, "randomization" passes at every washout, as its test of a constant effect is
exact there. At retention 0.5 no washout misses up to 0.12 and a washout of 2 passes at every
`rho` (0.0518 to 0.0565); one of 1 passes at `rho = 0.8` alone. At retention 0.8 no washout a
block of 4 allows is enough: the carryover's bias, not the error, sets the misses, and the block
must be longer. "hac" fails every cell.

The power, the share of tests reading "up": at an effect of 1 a period, `rho = 0.5`, retention
0.5 and a washout of 2, 0.387 by "randomization" (0.545 by "hac", whose size is 0.13); at an
effect of 0.5, `rho = 0.5`, retention 0.5 and a washout of 1, 0.140 (0.226). A washout trades
power for bias: without carryover and white noise the power at an effect of 1 falls from 0.872 to
0.816 and 0.725 as the washout leaves 1 and 2 periods of every block after a switch.

The default is "randomization", the method that holds its size on every cell. The tests read the
size of two cells over seeds 0 to 1999 (no lift and white noise; `rho = 0.8`, retention 0.5,
washout 2), "hac"'s over-rejection at `rho = 0.95` and washout 2, the bias's closed form, and,
over seeds 0 to 3999, the lag-0 bandwidth's size over 6 blocks at `rho = 0` without a washout
against Newey and West's over-rejection there.

## The burn-in, the jackknife and the order test (amended 2026-11-06)

Bojinov, Simchi-Levi and Zhao's design carries an order `m` of carryover, the periods after a
switch the previous state still reaches, a test of whether `m` is long enough, and an exact test
of the sharp null (the default reading above); Hu and Wager read a burn-in after each switch. Three
pieces were missing.

- **`read_switchback(..., burn_in=None)`.** `burn_in` periods of every block after a switch, and
  of the first block, are left out in place of the design's washout, of every arrangement the
  randomization reading reads, so its test stays exact. It is at least the washout, since a
  shorter one reads periods the design was planned to leave out, and below the block. A reading
  with a burn-in is the reading of the same design with that washout, by each method (a test
  holds it); `None`, the default, reads as before, bit for bit (a test holds that too). A burn-in
  is fixed before the KPI is read: one chosen after reading it, by the order test below say, gives
  the two steps together a size that is not the test's.
- **`method="jackknife"`**, the half-panel jackknife's error (Dhaene and Jochmans 2015, "Split-panel
  jackknife estimation of fixed-effect models", Review of Economic Studies 82). The blocks are cut
  into two halves in time, the first `ceil(B / 2)` and the last as many, an odd count's sharing
  its middle block; each half's difference is read over its own kept periods, kept as the whole
  test keeps them. The error is `|d_1 - d_2| / 2`, read as a Student t at 1 degree of freedom:
  exact where the halves' differences are independent normals of one variance and the whole's is
  their mean. Its 95 % quantile is `tan(19 pi / 40) = 12.706`
  (validation/switchback_carryover.mac, STEP 6, which also integrates the ratio's coverage to
  0.95). A half that keeps no period on or none off is refused, as are two halves that read the
  same difference. The lift is the other methods'.
- **`carryover_order(kpi, design, *, order=None)`**, the two-epoch test. In each block `g_b` is
  the period `order` after the block's start less the mean of the periods after it; the statistic
  is the mean of `g_b` over the blocks on after a switch less the mean over the blocks off after
  one (0 where a state has none), the carryover check at a washout of `order`, which defaults to
  the design's washout. Where the carryover reaches no further than `order` periods and the
  effect is one value over a block's periods past them, `g_b` is the same under every
  arrangement, so the statistic over the arrangements the randomization reading reads is its
  distribution under that null and the p-value is exact whatever the noise. A distributed lag of
  order `m` leaves every `g_b` at 0 read at order `m`, and one of order `m + 1` with weight
  `c_(m+1)` reads `-2 c_(m+1)` noise-free (validation, STEP 5, at orders 0 and 1).
  `CarryoverOrder` holds the order, the statistic, the p-value and the arrangements read.

### The evidence for the three

The jackknife's size at lift 0, the share of 95 % intervals that miss 0, on the world of the
evidence above (4000 tests a cell, seeds 0 to 3999, band [0.0397, 0.0603]); the tests refused,
where a half keeps no block on or none off, are left out (4 of 4000 over 12 blocks, 108 over 8):

| blocks | `rho` | washout 0 | washout 1 | washout 2 |
| --- | --- | --- | --- | --- |
| 12 | 0 | 0.0480 | 0.0408 | 0.0435 |
| 12 | 0.5 | 0.0470 | 0.0450 | 0.0450 |
| 12 | 0.8 | 0.0490 | 0.0513 | 0.0511 |
| 12 | 0.95 | 0.0563 | 0.0603 | 0.0593 |
| 8 | 0 | 0.0509 | 0.0491 | |
| 8 | 0.5 | 0.0439 | 0.0465 | |
| 8 | 0.8 | 0.0493 | 0.0496 | |
| 8 | 0.95 | 0.0540 | 0.0537 | |

Every cell is inside the band, `rho = 0.95` at washout 1 on its edge. On ADR 0058's null panels,
26 blocks of 4 with a washout of 1 and 200 draws a cell, it passes all six cells, 0.030 to 0.075
(validation/aa_audit.md). At an effect of 1 a period
it misses the lift, washout 0 / 1 / 2, 0.0673 / 0.0483 / 0.0475 at `rho = 0` and retention 0.5,
0.0581 / 0.0513 / 0.0478 at `rho = 0.5` and 0.5, and 0.1256 / 0.1014 / 0.0761 at `rho = 0` and
0.8, fewer than "randomization" (0.1245 / 0.0688 / 0.0528, 0.0960 / 0.0628 / 0.0518 and 0.4525 /
0.3033 / 0.1778) because its interval is wide, not because it reads the carryover. Its power is
low: 0.206 against "randomization"'s 0.872 at an effect of 1 without carryover or a washout,
0.184 and 0.167 against 0.816 and 0.725 at washouts 1 and 2, 0.108 against 0.387 at `rho = 0.5`,
retention 0.5 and washout 2, and 0.068 against 0.140 at an effect of 0.5 and washout 1. It reads
whether the two halves agree; the test is still "randomization".

The jackknife's bias correction, `2 d - (d_1 + d_2) / 2`, is not adopted as the lift: it takes
out the part of a carryover's bias that falls as `1 / B`, and the washout's part does not fall
with the blocks. Its mean against the difference's, as a share of the effect, over the
arrangements of 6 of 12 blocks of 4 whose halves keep both states (922 of 924), noise-free:
-0.2542 to -0.2355 at retention 0.5 and -0.6373 to -0.5994 at 0.8 with no washout (validation,
STEP 7), -0.1451 to -0.1294 and -0.5437 to -0.5018 with a washout of 1, and -0.0825 to -0.0687 and
-0.4604 to -0.4128 with a washout of 2; 8 and 20 blocks move alike. At retention 0.5 a washout of
1 alone takes the bias from -0.2542 to -0.1451, the correction alone to -0.2355.

The order test on the world of the tests: a level of 10, an AR(1) noise of variance 1 at `rho`,
blocks of 4 with a washout of 1, the channel's state through a distributed lag summing to 1, read
at order 1. Its size where the carryover is of order 1 (weights 1/2, 1/2, effect 2) over 2000
designs and noises (seeds 0 to 1999, the binomial standard error of 0.05 is 0.0049, the band
0.05 +- 0.02 is over 4 of them), and its power where it is of order 2 (weights 1/3 each, so
`effect / 3` reaches the second period after a switch) over 1000 (seeds 0 to 999, standard error
at most 0.016). 12 blocks enumerate their 924 arrangements; 24 read 10 000 drawn:

| blocks | `rho` | size | power, effect 1 | effect 2 | effect 3 | effect 4 |
| --- | --- | --- | --- | --- | --- | --- |
| 12 | 0 | 0.0570 | 0.074 | 0.168 | 0.305 | 0.500 |
| 12 | 0.5 | 0.0565 | 0.077 | 0.214 | 0.424 | 0.635 |
| 12 | 0.8 | 0.0550 | 0.115 | 0.408 | 0.723 | 0.926 |
| 24 | 0 | 0.0535 | 0.132 | 0.375 | 0.706 | 0.914 |
| 24 | 0.5 | 0.0550 | 0.167 | 0.518 | 0.843 | 0.979 |
| 24 | 0.8 | 0.0500 | 0.304 | 0.838 | 0.997 | 1.000 |

The size is exact, not estimated: under the null each block's gap is the same under every
arrangement, so whatever the noise exactly 46 of the 924 arrangements of 12 blocks reject at 5 %
taken as the one run, 0.0498 (a test holds it). The six cells share their seeds and so their
Monte Carlo errors, 0.0500 to 0.0570, within 1.5 standard errors of it. The power reaches 0.8 over
24 blocks at an effect of 4, the second period's carryover 1.33 noise standard deviations, at
every `rho` (0.914 to 1.000), and at 3 where `rho >= 0.5`; over 12 blocks, half as many blocks
after a switch, only at `rho = 0.8` and an effect of 4. A noise that persists raises it: the
two epochs are adjacent periods, whose shared noise the gap takes out. The tests hold the size of
order-1 worlds at `rho = 0.5` over 12 blocks and 2000 designs to 0.05 +- 0.02, and the power
over 500 order-2 worlds of 24 blocks at an effect of 4 and `rho = 0` to more than 0.8 by three
standard errors.

## Consequences

- A national channel can be tested and read inside mixdyn, and its estimate calibrates a model as
  a geo test's does: the lift over the test's periods against the spend as planned.
- The design must be planned around the channel's carryover: a washout long enough that the
  adstock left from the previous block is small, `lam^(washout + 1)` of the switch, and blocks
  long enough to keep periods after it. A washout leaves periods out and costs power; the
  carryover check flags a washout too short, but is not a test of it.
- The randomization reading needs enough arrangements: at least 8 blocks with half on for its 95 %
  interval to be bounded. Its interval is the connected set of effects about the estimate the test
  does not reject. The accepted effects have no hole (amended 2026-10-31): each arrangement's
  slope in the effect, its share of periods on among its kept periods on less that among its kept
  periods off, lies in `[-1, 1]`, so once an arrangement stops being as far from 0 as the observed
  one it stays so, and the test rejects every effect past the interval's ends. The p-value against
  0 and the interval's verdict therefore agree, but where an end falls on 0 itself, which a KPI
  of whole numbers can make happen and rounding then sets the verdict of.
- **Not covered:** a carryover that outlasts the washout, which the reading reads as part of the
  periods off's or on's level and so biases (the tables at retention 0.8); interference between
  periods of any other kind (an effect that builds over the test, or a demand shifted from the
  periods on into the periods off, which the difference reads twice); an effect that is not one
  constant a period, which the randomization interval assumes, though its p-value against no
  effect does not; and a trend. Over a design's arrangements each block is on as often as off at
  half the blocks on, so a linear trend moves the difference by 0 on average, exactly
  (validated: the mean over the 924 arrangements of a trend of 1 a period is 0 to 1e-15 at every
  washout); it does not bias the reading and the randomization test stays exact under it, but it
  adds to the difference's spread (a standard deviation of 8.3 to 10.1 KPI a period for a trend
  of 1 a period over 48 periods), which costs power. The reading does not detrend.
- *On null panels* (amended 2026-11-06). ADR 0058's audit (`validation/aa_audit.md`: 200 null
  panels of 104 weeks a cell, the false-positive rate at a nominal 0.05, the band [0.03, 0.08])
  reads the national KPI of the same panels, 26 blocks of 4 weeks, a washout of 1, half the
  blocks on, by each `Method`. Under AR(1) noise, a trend and a shift, at 10 and 30 geos:
  "randomization" 0.065 and 0.030, 0.030 and 0.040, 0.060 and 0.045, in the band in all six;
  "hac" 0.125 and 0.090, 0.100 and 0.090, 0.105 and 0.080, above it in five, the reporting
  fallback without a size this ADR names it.

## Alternatives

- **The Newey-West reading as the default**: rejected; it over-rejects on every cell above, 0.086
  to 0.164, with 12 or 8 blocks and at any `rho`. It stays selectable by `method="hac"` for a
  design too small to enumerate, as a reporting fallback without a size.
- **The lag-0 bandwidth as "hac"'s default**: not taken (amended 2026-10-31); it holds its size
  on 15 of the 18 cells of its table and fails 3, so it does not make "hac" a test, and the
  default bandwidth's readings stay as they were.
- **A washout on every block**, switched or not: rejected; a block in the state of the one before
  carries over its own state, and leaving its first periods out only costs power.
- **Independent draws of each block's state** (a Bernoulli design, Bojinov et al.'s simplest): it
  can leave a state with one block or none, and the re-randomisation distribution is the design's
  only if it is the one drawn from; a balanced design keeps both states at a fixed count.
- **Regression adjustment for carryover**, the previous block's state as a covariate: it assumes
  the carryover's shape, the very thing the washout avoids assuming, and adds a parameter to few
  blocks. The closed form and the carryover check serve as diagnostics instead.
- **The half-panel jackknife's corrected lift** (amended 2026-11-06): not adopted, above; it takes
  0.014 to 0.048 of the effect off a carryover's bias, a washout of 1 at retention 0.5 0.109.
- **The jackknife as the default error** (amended 2026-11-06): it holds its size on every cell,
  but at 1 degree of freedom its power is a quarter to a half of the randomization test's.
- **Detrending before reading**: rejected for now; the design's balance keeps the trend out of the
  mean of the difference, and a fitted trend would make the randomization reading depend on the
  fit.
