# 0077. The calibration loop as a scorecard arm

Status: accepted, 2026-11-07; adds the arm `voi` to ADR 0021's harness, its three variants beside
the static arm's; no existing arm, record or output changes

## Context

The engine values a lift test before it runs (`mixdyn.experiments.value_of_tests`, ADRs 0030
and 0032) and chooses the set worth running (`choose_tests`, ADR 0031). Nothing scored that loop:
no arm read a test it had chosen, refitted with it and planned again, so whether the value
predicts what a test is worth to the plan was never measured on the bench's worlds. Meridian's
documentation recommends a test by another rule, a channel-calibration score built from the
posterior's returns alone ("Channel calibration recommendation", updated 2026-09-11). The two
choices are compared at rung 1, each by its regret after the refit.

The bench's ladder (`causaldyn_bench.scorecard.ladder`) gives rung `k` each channel's latest `k`
tests. Rung `k` holds rung `k - 1`'s tests and each channel's earliest one more, every test read
in a universe of its own, so a test's reading is the same at every rung that holds it; its
standard error is read from the pre-test weeks of every test of its channel at the rung, so the
rung below's rows can differ in their errors alone. The bench scores one weekly spend a channel.

## Decision

- **The arm `voi`** (`scorecard.arms.voi`) reads an export at rung `k` as the loop's next step
  from rung `k - 1`: every row but each channel's earliest is a test already run, and each
  channel's earliest a candidate. The export holds every candidate's reading; the arm reads only
  the chosen one's.
- **Before, values, refit.** The static arm's two fits (`scorecard.arms.bayes`) of the tests run,
  stacked over every draw with their leave-one-out densities kept; `value_of_tests` on that
  posterior, `OUTER` outer draws (`SMOKE_OUTER` in a smoke run), each reweighted by the test's
  likelihood, plans read on a grid of `GRID` spends, from a child of the export's seed; then the
  static fits again with the chosen test's row, and the static arm's `stacked` plan of them.
- **The choice is the candidate of the largest reduction**, the first on a tie. `choose_tests` is
  asked beside it, at a revenue of 1 a unit of the KPI and one use, and its set and verdict are
  recorded with whether it agrees. CHOICE: `choose_tests` ranks by net value, the reduction less
  the spend a test saves and plus the sales it loses. The bench's tests ran in universes of their
  own, so the quarter it scores pays for none of them, and the regret after the refit rewards the
  reduction alone.
- **A candidate is its row's test in a share of the market.** A whole-market go-dark test sits
  at `choose_tests`' share limit of a half. Each candidate is the row's test at share `s`, its
  spend, lift and error `s` times the row's, which the model reads as the same likelihood
  whatever `s` (`LiftEstimate` reads a group's spend and lift over its share): its reduction does
  not move with `s`. The shares price every candidate at one gross cost, half the cheapest's
  whole-market go-dark, the spend it adds and the sales the posterior expects it to lose, so
  `choose_tests` affords one test at a time.
- **Meridian's score**, each channel's `0.5 implausible + 0.25 variance + 0.25 bias` in points of
  100, read on the posterior before: a bowl in the log of the mean return about the geometric mean
  of the bounds 0.5 and 20, its half-width `log(sqrt(40) / share)`; a half-bowl in the 80 %
  highest-density interval's width over the median, over the default prior's 2.0687, times the
  channel's share of the spend, 100 up to 0.5 and 50 at 1; and `100 b^(log 0.5 / log 0.1)` of
  the largest absolute correlation `b` of the channel's spend with a control, 0 without one. The
  lowest channel under 67.5 is recommended. Maxima holds the prior's width ratio, 2.068704, and
  the bowls' anchors (`validation/meridian_calibration_score.mac`).
- **Three variants**, each the static arm's `stacked` plan of its posterior and its record: `voi`,
  the test of the largest reduction; `voi meridian`, the recommended channel's candidate, or no
  test where none scores under 67.5; `voi no test`, the posterior before. At rung 1 `voi` and
  `voi meridian` compare the two choices of the first test; at every rung `voi no test`'s regret
  less `voi`'s is the reduction realised. At rung 0 no test is left, and every variant plans the
  posterior before. A geo export's market is refused: its tests are read in a share of it
  already.
- **The static arm's fit is shared**: `bayes.fitted` makes its record from a pair of fits, so the
  loop's three fits are the static arm's, sampled as it samples.

## Consequences

- A world costs the static arm's fit twice, three times where the two choices differ, and one
  `value_of_tests` and one `choose_tests` call.
- Each record carries every candidate's share, reduction and its Monte Carlo error, each channel's
  calibration score and its parts, the test cost, the expected value of perfect information, the
  chosen test's reduction, prediction and interval, and `choose_tests`' set and verdict. Whether
  the value predicts the realised reduction is read across worlds from these beside the regrets.
- The loop reads one test a rung. A loop of several steps, each refit choosing again, is a run
  over rungs, read from the records.
- Meridian's score is restated from its documentation, not run from its code. Where its
  documentation and its code disagree, the record follows the documentation.

## Alternatives considered

- **`choose_tests`' set as the choice.** Rejected for the scored plan: it favours a test whose
  go-dark saves spend on a channel that returns little, which the bench does not credit. It is
  recorded beside the choice.
- **Candidates at share 1.** Rejected: `choose_tests` refuses a set above its share limit, so a
  whole-market test is never affordable.
- **A candidate's reading drawn from the posterior.** Rejected: the world's own reading is in the
  export; the loop reads it, as a test read in practice would be.
- **Meridian's score by its code.** Not available here: the engine is Python only and adds no
  dependency, and Meridian is the bench's tool, not the engine's.
