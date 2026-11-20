# 0058. Readers audited on null panels

Status: accepted, 2026-10-31; audits ADR 0047's geo readers (with ADR 0045's placebo-window error
they share), ADR 0048's synthetic difference in differences and ADR 0056's switchback on panels
with no effect, and adds opt-in randomisation readings for the readers that fail a cell; every
default is unchanged

## Context

A reader's interval feeds the model it calibrates: a test read too tightly moves every arm the
lift reaches, and one read too loosely spends the test's power. Each reader was checked when it
was written, on the panel its ADR chose: ADR 0047's on 10 geos of 52 weeks at an autoregression of
0.8, ADR 0048's on 20 controls of 15 weeks, ADR 0056's on a national KPI of 12 blocks. Three
published findings say the readers should also be read where their assumptions bend:

- Google's GeoX A/A study (2026-09) found a geo test's t-test rejecting a true null above 10 times
  its level under an AR(1) of 0.5, and sliding-window placebos overconfident under a trend or a
  break.
- A public synthetic difference-in-differences harness reads its placebo errors as "slightly
  optimistic" where the donor pool is thin.
- A review of ADR 0056's switchback found its `hac` method rejecting 1.6 to 2.7 times its level.

## Decision

- **One harness, `validation/aa_audit.py`, reads every reader on the same null panels.** 104
  weeks, 10 or 30 geos, no treatment effect; each geo's level log-normal about 100, a seasonality
  all share, and a noise of `0.05` of its level times a unit series of one of three laws: an AR(1)
  of 0.5; the same with a linear trend of its own a geo (a drift of the order of its noise a
  year); the same with a level shift of 2 all geos share from a uniformly drawn week. Each reader
  reads each panel at its shipped defaults on a null treatment its own design draws: the geo
  readers on `assign(pair_geos(...))`'s matched pairs over the last 8 weeks, `read_sdid` on 30 %
  of the geos drawn at random over the same weeks with every other geo a donor, and
  `read_switchback` at each of its methods on the summed KPI in 26 blocks of 4, a washout of 1.
  `numpy.random.SeedSequence(20261031)` spawns one child a cell of noise and geos, which spawns
  one a draw, so every reader reads the same 200 panels and designs a cell.
- **The band.** A cell's false-positive rate is the share of 200 draws whose p-value is below 0.05,
  its Monte Carlo error 0.0154, and its coverage the share whose 95 % interval holds 0. It passes
  within [0.03, 0.08]: the top is 0.05 plus two errors (0.0808), rounded; the floor is tighter
  than two errors (0.0192), so that a reader far under its level fails too. A reader at the level
  passes a cell with probability 0.914, so about 3 of 36 cells fail by chance, and one at 0.10
  passes with 0.207. A cell that fails at the edge is read again over 1000 draws before a reader
  is judged on it.
- **The table is `validation/aa_audit.md`**, with the command that regenerates it and its date;
  `tests/test_aa_audit.py` reads 3 of its cells over their first 60 draws, within 3 Monte Carlo
  errors (0.084) of 0.05, and pins their counts of rejections and the harness's draws.
- **A reader that fails a cell gets an opt-in alternative**, its defaults unchanged, read on the
  same panels:
  - **Design-aware unit placebos** for the geo readers, `read_did(..., variance="unit_placebo",
    pairs=...)` and `read_tbr(...)` alike, and for synthetic difference in differences,
    `read_sdid(..., variance="unit_placebo")`. The placebo treatment sets are drawn by the
    test's own design rule: every assignment of the matched pairs, each pair swapped or not
    (enumerated to 100 000, otherwise 10 000 drawn beside the observed one), or every `N_tr` of
    all the geos (enumerated to `replications`, otherwise that many drawn). Each lift, the
    observed one among them, is studentised by its pre-period out-of-sample error: the
    estimator fitted on the pre-period less its last readout's length, the root mean square of
    its per-period gaps over those held-out weeks. The p-value is the share of studentised lifts
    at least as far from 0 as the observed one; the interval is the lift plus or minus its own
    error times the `ceil(alpha A)`-th largest of the `A` in absolute value, so that it holds 0
    exactly where the p-value is at least `alpha`. With no effect the test is exact whatever the
    noise: the KPI is held fixed and only the assignment moves. It was written from GeoX's
    description of its rule, without reading its code.
  - **Pair sign-flip randomisation inference**, `read_pairs(panel, design, pairs)`, for the
    matched-pair reading: each pair's gap, treated less control, less its pre-period mean,
    summed over the readout, is the pair's contribution, the lift their sum. Swapping a pair
    flips its contribution's sign exactly, so the lift's distribution over the design is that of
    `sum_p s_p c_p` over the signs. The interval inverts the same test under a lift shared alike
    by the pairs; each assignment is as far from 0 as the observed one on one interval about the
    estimate, of end `|a'| / (1 +- sign(a') b)` (`a'` the assignment's lift less its mean sign
    `b` times the observed one), so the ends are exact; `validation/sign_flip_interval.mac`
    checks the end and, over every sign of 4 pairs, the count of assignments as far at 161
    points either side, and a test runs it where `maxima` is installed. Fewer than 6 pairs are refused: a sign
    and its reverse read alike, and 5 pairs' least p-value is 2 / 32.
  - Each hands on a `LiftEstimate` whose error is the farther of its 95 % interval's ends from
    the lift over `z(0.975)`, as the switchback's randomisation reading does, and its reading's
    error is the spread of the lifts over the assignments, at an infinite count.
- **The switchback is audited only.** Its module and tests are another change's; the table reads
  it at every method `switchback.Method` names, and a method added there is read by the next run.

### The evidence

At the defaults, over 200 draws a cell (the table holds every cell with its coverage):

| reader | ar1, 10 | ar1, 30 | trend, 10 | trend, 30 | shift, 10 | shift, 30 |
| --- | --- | --- | --- | --- | --- | --- |
| `read_did` | **0.025** | 0.055 | 0.065 | 0.065 | 0.065 | 0.045 |
| `read_tbr` | **0.010** | **0.015** | **0.010** | **0.020** | **0.015** | **0.025** |
| `read_synthetic_control` | 0.035 | 0.050 | 0.045 | 0.055 | 0.040 | 0.030 |
| `read_sdid` | 0.055 | 0.050 | **0.020** | 0.035 | 0.030 | 0.030 |
| switchback, randomization | 0.065 | 0.030 | 0.030 | 0.040 | 0.060 | 0.045 |
| switchback, `hac` | **0.125** | **0.090** | **0.100** | **0.090** | **0.105** | 0.080 |

No reader at its default over-rejects but the switchback's `hac`, 0.080 to 0.125, 1.6 to 2.5
times its level, as ADR 0056 found. `read_tbr` is conservative in every cell (0.015 over 1000 draws at AR(1) and 30 geos): its
placebo windows refit the regression on fewer periods than the test does, so their lifts carry
more of the parameters' error than the test's. `read_did`'s one failing cell reads 0.036 over 1000
draws, within the band; `read_sdid`'s at a trend and 10 geos is under its level.

The alternatives, on the same draws:

| reader | ar1, 10 | ar1, 30 | trend, 10 | trend, 30 | shift, 10 | shift, 30 |
| --- | --- | --- | --- | --- | --- | --- |
| `read_did`, unit placebos | 0.030 | 0.070 | 0.030 | 0.045 | 0.035 | 0.040 |
| `read_tbr`, unit placebos | **0.015** | **0.085** | 0.030 | 0.050 | 0.045 | 0.045 |
| `read_pairs` | refused | 0.070 | refused | 0.030 | refused | 0.045 |
| `read_sdid`, unit placebos | 0.040 | 0.030 | 0.045 | 0.045 | 0.040 | 0.050 |

`read_tbr`'s unit placebos fail two cells, both read again over 1000 draws: 0.055 at 30 geos,
where the test is exact and 0.085 was the draws' (the AR(1) cell at 30 geos reads high in every
exact reading on its 200 panels: 0.070 for the other two); and 0.029 at 10 geos, where 5 pairs'
32 assignments let a test reject at most `1/32 = 0.031` of the time: at 10 geos the matched-pair
randomisation readings sit at the band's floor by construction. The tests check exactness
directly: over every assignment of one panel's 6 pairs at most `ceil(0.05 x 64) - 1 = 3` reject,
for each of the three pair readings, and over each of 21 geos treated at most 1 for synthetic
difference in differences.

## Consequences

- Each reader's size under an AR(1), a trend and a break is on record, and a change to a reader
  that moves it moves `tests/test_aa_audit.py`'s pinned counts first.
- `read_tbr`'s default reads too wide an interval; `variance="unit_placebo"` holds its level. The
  default stays: a new default ships opt-in first.
- The randomisation readings need the design's pairs, which `GeoTest` does not hold, and a test
  designed as `assign` designs it; a test whose treated geos were chosen, not drawn, has no
  randomisation to read.
- Under a real lift, unit placebos that swap a pair or treat a treated geo read part of the lift
  in their spread, so their interval is wider than the lift's own spread and the test's power is
  less than the default's. The sign flip has no such cost; its interval assumes a lift shared
  alike by the pairs.
- At 10 geos, 5 pairs: the sign flip is refused, and unit placebos reject at most 1/32.
- **Not covered:** a noise that differs between the treated and the control geos, spillover, a
  lift that is not shared alike, panels shorter than 104 weeks, block designs of more than two
  geos (the pair readings refuse them), the readers' power, and the switchback under carryover
  (ADR 0056's tables).

## Alternatives

- **Unit placebos for synthetic difference in differences drawn among the controls alone**, as
  the paper's Algorithm 4 draws its placebos, studentised the same way: rejected. On the same
  draws it read 0.070, 0.050, 0.085, 0.050, 0.060, 0.085 (failing two cells, 0.067 over the
  1200): the observed treated set has every control as a donor, a placebo set only the other
  controls, so the two are not exchangeable and the studentised placebos understate the
  observed one's spread. Drawn by the design rule over all geos, the observed set is one of the
  sets the design could have drawn, and the test is exact.
- **A sign flip of `read_did`'s own lift**: not possible; its ratio of the groups' pre-period
  means makes a swap of one pair change the lift other than by a sign. `read_pairs` reads each
  pair against its own pre-period instead.
- **The band at 0.05 plus or minus two errors, [0.0192, 0.0808]**: wider at its floor; it would
  pass `read_tbr`'s default in two cells it now fails, which reads a test's power away
  unnoticed.
- **A studentised interval inverted under a lift**, every assignment re-read at each candidate
  lift: exact under a shared lift, as the sign flip's is, but its cost is a read of every
  assignment per candidate; kept as the sign flip's, where the inversion is in closed form.
