# 0037. A lift read over the panel's own cells

Status: accepted, 2026-10-07

## Context

ADR 0015 reads a lift test's summary, its lift over the readout and the lift's standard error, as
a term of the likelihood, and ADR 0018 names the geos the test ran in. Neither places the test in
the panel's calendar. A test run in the market the panel holds reads the panel's own cells: its
analysis, a synthetic control or a difference in differences, sets the tested geos' KPI over the
readout against a counterfactual. Where the model reads those cells too, the likelihood may hold
two readings of one noise, and the posterior may come out too narrow, or pulled toward the test.

Which cells share the lift's noise follows from where its control lies. The lift's error is the
tested group's noise over the readout less the counterfactual's error.

- *A group that is the whole of its geos' market* (`share` 1) is read against a control outside
  them: other geos, another market, a forecast. Its geos' cells over the readout hold its noise,
  all of it.
- *A group of a share `s` of its geos' market* is read against the rest of that market scaled to
  its size, as ADR 0030 reads a test. Its error is `n_g - s / (1 - s) n_r`, the cells hold
  `n_g + n_r`, and with a period's noise variance in proportion to the people it is summed over
  (ADR 0013) the two covary by `s v - s v = 0`.
- *A control's cells and a pre-period's* enter the lift's error with the other sign. A likelihood
  that reads both is wider than it should be, not narrower.

What a shared noise can cost has a bound. Let the lift's error correlate with the panel's reading
of the channel by `rho`, and let the lift hold a share `w` of the channel's precision. In a normal
model with a flat prior, the posterior mean then varies over the noise by `1 + 2 rho sqrt(w (1 -
w))` times the posterior's variance. That is at most 2, at `rho` 1 and `w` 1/2. `rho` is at most
the correlation of the lift's error with the noise of one readout cell.

PyMC-Marketing's lift measurements carry no dates; its lift-test notebook allows that a test's
spend may already be in the training data, and says no more. Meridian turns a test into a prior on
the return and fits every week of its data. Robyn refuses an experiment outside its modelling
window, so its fit always reads the experiment's weeks, and `MAPE.LIFT` compares its decomposition
over them with the lift. None of their documents speaks of counting those weeks twice.

The bench's ladder runs each test in universes of its own, so the history an arm fits holds none
of it: no rung overlaps. Each universe adds noise of a share of mean weekly sales, of opposite
signs in the two; the world's own noise is common to both and cancels in the gap. The overlapping
rung, `causaldyn_bench.scorecard.ladder.overlap`, runs a rung's tests of one channel in a share of
the market whose history the model reads. At share 1 the history is the tested universe. It holds
the tests' spend and response, and in every week the analysis noise, which each lift holds twice
over its readout. One readout cell then shares 0.6 % of its residual variance, as the fits read
it, with the lift at an analysis noise of 1 % of mean sales, and 37 % at 10 %. At share 1/2 the
history is the two halves' mean: the tests' spend and half their response, and none of their
noise.

It was measured on Track M v2's drawn worlds 900 to 929, at rung 4: four go-dark tests of paid
shopping, dark from weeks 20, 55, 100 and 140 for 4 weeks and read over those and the 8 after, 48
of the 156 weeks. Each world is fitted by the one-geo model with the world's curve (tanh), the
export's controls, a trend of 3 bends and 3 harmonics of the year: 4 chains of 500 warm-up draws
and 500 draws, from a seed spawned from the world's. Each fit is planned from 400 of its draws at
the quarter's budget, and its regret per euro of that budget read against the oracle's plan. None
of the 540 fits diverged; the largest R-hat was 1.035. The bench reads a test only where its sales
fell (`lift_rows`): all four at 1 %, 2.9 of 4 on average at 10 %. A `LiftEstimate` keeps every
test, as the third block does. The variants:

- *as today*: the history as planned, the tests in universes of their own;
- *(a) both*: share 1, every cell and every lift;
- *(b) cells out*: as (a), the readouts' KPI missing (ADR 0002), their spend kept;
- *(c) no lift*: as (a), without the lifts;
- *(i) noise apart*: as (a), each lift read from the same tests with the analysis noise drawn
  again, from the world's seed plus 100 000. It is (a) without the double count.

Paid shopping's return against the world's, with Monte Carlo errors over the worlds; a coverage's
is about 0.08:

| analysis noise, tests read | variant | error / truth | SD / truth | mean z² | 80 % / 90 % | regret per euro |
|---|---|---|---|---|---|---|
| 1 %, the bench's | as today | +0.006 ± 0.020 | 0.115 | 1.05 ± 0.22 | 0.73 / 0.90 | 0.173 ± 0.033 |
| | (a) both | +0.031 ± 0.020 | 0.125 | 0.95 ± 0.18 | 0.70 / 0.93 | 0.173 ± 0.034 |
| | (b) cells out | +0.004 ± 0.021 | 0.124 | 0.95 ± 0.20 | 0.77 / 0.90 | 0.136 ± 0.024 |
| | (c) no lift | +0.588 ± 0.140 | 0.581 | 2.43 ± 0.70 | 0.67 / 0.73 | 0.188 ± 0.037 |
| | (i) noise apart | +0.015 ± 0.015 | 0.113 | 0.51 ± 0.11 | 0.93 / 1.00 | 0.179 ± 0.034 |
| | as today, cells out | +0.003 ± 0.021 | 0.122 | 0.96 ± 0.20 | 0.70 / 0.90 | 0.135 ± 0.024 |
| 10 %, the bench's | as today | +1.135 ± 0.175 | 0.771 | 3.56 ± 0.87 | 0.47 / 0.70 | 0.138 ± 0.026 |
| | (a) both | +0.421 ± 0.112 | 0.581 | 1.49 ± 0.47 | 0.80 / 0.83 | 0.158 ± 0.027 |
| | (b) cells out | +0.313 ± 0.115 | 0.714 | 0.90 ± 0.23 | 0.87 / 0.93 | 0.140 ± 0.025 |
| | (c) no lift | +0.342 ± 0.128 | 0.607 | 1.44 ± 0.47 | 0.77 / 0.90 | 0.173 ± 0.031 |
| | (i) noise apart | +0.525 ± 0.125 | 0.611 | 1.74 ± 0.61 | 0.80 / 0.87 | 0.159 ± 0.029 |
| 10 %, every test | as today | +0.583 ± 0.145 | 0.664 | 1.96 ± 0.58 | 0.67 / 0.83 | 0.155 ± 0.029 |
| | (a) both | +0.181 ± 0.094 | 0.509 | 1.27 ± 0.36 | 0.77 / 0.90 | 0.167 ± 0.027 |
| | (b) cells out | −0.095 ± 0.089 | 0.564 | 1.21 ± 0.30 | 0.77 / 0.90 | 0.128 ± 0.023 |
| | (i) noise apart | +0.167 ± 0.088 | 0.510 | 1.17 ± 0.55 | 0.90 / 0.97 | 0.165 ± 0.029 |
| | as today, cells out | +0.069 ± 0.108 | 0.607 | 1.22 ± 0.30 | 0.80 / 0.83 | 0.157 ± 0.028 |
| | share 1/2, (a) both | +0.606 ± 0.132 | 0.640 | 2.11 ± 0.55 | 0.70 / 0.73 | 0.191 ± 0.031 |
| | share 1/2, (b) cells out | +0.077 ± 0.109 | 0.611 | 1.17 ± 0.29 | 0.80 / 0.83 | 0.155 ± 0.028 |

`z` is the error over the posterior's SD; a lift counted twice raises its mean square above (i)'s
at the same SD. Paired on the same worlds: an SD's ratio is the geometric mean with two standard
errors, and a mean z²'s ratio has a 95 % percentile bootstrap over the worlds (20 000 resamples,
seed 20261007).

- *The double count*, (a) against (i): mean z² 0.86 times (i)'s [0.67, 1.20] at 10 % on the
  bench's reading, and 1.08 times [0.77, 2.11] on every test. The 80 % coverage moves by
  +0.00 ± 0.07 and −0.13 ± 0.09, the absolute error by −0.070 ± 0.058 and +0.064 ± 0.065 of the
  truth, the regret by −0.001 ± 0.013 and +0.001 ± 0.014. At 1 % the pair holds nothing: the
  lifts hold nearly all the precision and (i) draws them again, so the two z's correlate at −0.04.
  There (b) against (a), on the same lifts, reads it: 1.00 [0.79, 1.23].
- *The bound*: `rho` is at most 0.08 at 1 %, where `w` is about 0.95, so the bound is 1.03. It is
  at most 0.61 at 10 %, where `w` is about 0.3 on every test, so the bound is 1.55. The
  measurement does not exclude it there; its point estimate is 1.08.
- *Taking the cells out*, (b) against (a): the SD 1.002 times (a)'s [0.963, 1.043] at 1 %, 1.209
  times [1.107, 1.319] and 1.077 times [0.970, 1.196] at 10 %. Mean z² 1.00, 0.60 [0.29, 1.22]
  and 0.95 [0.44, 1.92] times (a)'s. The mean moves by −0.20 ± 0.04, −0.18 ± 0.18 and −0.52 ±
  0.18 of (a)'s SD, the regret by −0.037 ± 0.028, −0.019 ± 0.021 and −0.038 ± 0.021.
- *The same weeks without a test*, as today with its cells out against as today. At 1 % the mean
  moves by −0.02 ± 0.02 SD, and the regret by −0.038 ± 0.031, as much as in (b). At 10 % on every
  test the mean moves by −0.75 ± 0.11 SD and the absolute error by −0.28 ± 0.11 of the truth; the
  regret, by +0.002 ± 0.008. The third readout, 25 November 2024 to 10 February 2025, is a holiday
  season. Sales there run at 1.21 times their mean (1.09 to 1.34 over the worlds), and every
  channel's spend rises with them: paid shopping's 1.31 times, meta's 1.14 and television's 2.45.
  At 1 %, taking the readouts out moves meta's return from +1.23 ± 0.36 to −0.01 ± 0.08 of its
  truth with the tests in those weeks, and from +1.24 ± 0.38 to −0.01 ± 0.08 without them.
- *At share 1/2*, where no noise is shared, (b) against (a) on every test: the mean moves by −0.80
  ± 0.14 SD and the absolute error by −0.26 ± 0.12 of the truth, as the same weeks move it without
  a test. The regret moves by −0.036 ± 0.021. Against as today, (a) plans worse by +0.037 ± 0.022
  and (b) as well as it, +0.000 ± 0.008.
- *The regret at 10 % on every test*: a history that holds the tests plans worse than one that does
  not, unless their readouts are out. (b) gains 0.036 ± 0.021 over (a) at share 1/2 and 0.038 ±
  0.021 at share 1, where the same weeks without a test gain nothing. The lifts' noise has no part
  in it: (i) plans as (a) does. It is not established at 30 worlds, and paid shopping's return
  does not show why.
- *Dropping the lift*, (c) against (a): at 1 % the absolute error grows by 0.61 ± 0.12 of the truth
  and the SD 4.7 times. At 10 % the lifts weigh little: the regret grows by 0.015 ± 0.011.

## Decision

- **A lift and the cells its readout reads both stay in the likelihood.** `LiftEstimate` gains no
  place in the calendar, and the models neither refuse an overlap nor take cells out. The same
  holds for a geo test read period by period (`chc.lift.LiftTest`), which was not measured.
  ADR 0038: `LiftEstimate.end` places a test, for the DLM, which reads the coefficients over its
  readout; the cells stay.
- The measurement found no double count to act on. Where the lifts weigh most, at 1 %, the cells
  share almost none of their noise. Where the cells share more, at 10 %, the lifts weigh little,
  and (a) is no worse calibrated than (i) and plans as (i) does. What taking the cells out moved
  in paid shopping's return came with the weeks, not with the tests.
- A user who wants a readout's cells out sets the KPI missing there. The models read a missing KPI
  as an unobserved cell and keep its spend for the carryover (ADR 0002), which is what (b) ran.
- `scripts/study_lift_overlap.py` runs the measurement again, with the bench on the path.
- **Revisit** in two cases. A test that holds about half of its channel's precision, read against
  a control outside the panel's market, at an analysis noise that is most of its cells' noise:
  there `rho` nears 1 and the posterior's variance can be half of what it should be, and the bench
  has no rung there yet. Or a larger run that establishes the regret a history holding the tests
  loses at 10 %: the question then is why the model misreads a go-dark in its own history, before
  any cell leaves the likelihood.

## Consequences

- Nothing in the API or the models changes. A test not yet run (`Candidate.estimate`) and a test
  run elsewhere read as before.
- A posterior read with a test in the panel's weeks can be too narrow by the bound above, a factor
  of at most `sqrt(2)` on its SD. On the bench it was not.
- Taking weeks out moves a fit through their calendar as well as through their tests. A user who
  takes a readout out by hand can move the fit for that reason alone, as the holiday season moved
  meta's return and paid shopping's here.
- The overlapping rung lives in the bench's ladder; the scorecard's families do not draw it yet.

## Alternatives considered

- **Take a readout's cells out of the likelihood**, each estimate placed by the date of its
  readout's last period: a whole group's, whose noise the cells share, or every group's. Built for
  whole groups and tested, then set aside. It costs up to 21 % of the return's SD at 10 %, and
  recovered no calibration that (i) shows lost. Its pull on paid shopping's return is the weeks',
  which pull it as far with no test in them. Its gain in regret at 10 % is not established, and is
  not the double count's.
- **Refuse an overlap**, naming the cells. Rejected: it needs the same place in the calendar, and
  leaves the user (b) or (c). Dropping the lift, (c), costs most where the lifts carry the
  channel, at 1 %.
- **Widen the lift's error** by the share of it the cells hold. Rejected: that share is the shared
  noise's variance over the lift's, which a summary does not report.
- **Model the covariance**, the lift and its cells read as one multivariate normal. Rejected for
  the same reason, and the cells' residuals hold misfit that the test's analysis never saw.
