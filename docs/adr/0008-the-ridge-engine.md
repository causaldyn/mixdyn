# 0008. The ridge engine, in Robyn's style, calibrated by geo tests

Status: accepted, 2026-10-02

## Context

Robyn (Runge et al. 2024) is the static media-mix model most teams start from: each channel's spend
carried over, saturated by a Hill, and regressed with the controls by ridge, its hyperparameters
searched by Nevergrad over in-sample NRMSE and DECOMP.RSSD, and a model picked from the Pareto
front's clusters. The engine needs that model from a `MediaPanel`, in Robyn's conventions so that a
model ported from Robyn reads the same, and two things Robyn does not give: a choice that predicts,
and the geo tests in the fit rather than beside it. It runs as R, which a Python engine cannot
call, and glmnet, Robyn and prophet are oracles, never dependencies (ADR 0001).

## Decision

- **`ridge_path` is glmnet's bounded ridge, solved exactly.** One SVD of the standardised design
  gives every penalty's free solution in closed form; where a bound binds, bounded-variable least
  squares on the augmented system ends at the Karush–Kuhn–Tucker point of a strictly convex problem.
  The penalty has no units: glmnet's `lambda` is it times the KPI's standard deviation. A value
  within `1e-12` of the largest standardised coefficient from its bound is on it.
- **The transforms are Robyn's, on `chc.response`'s kernels and curve.** A geometric theta or a
  Weibull CDF's shape and scale, then a Hill whose half-saturation point is `gamma` times the most
  of the window's adstocked spend (Robyn 3.12). A Weibull scale is kept in periods, so a longer
  series moves no earlier value; `robyn_weibull` maps Robyn's share of the series onto it. Robyn's
  min-max-scaled Weibull PDF kernel is not offered: which of its weights the scaling sets to 0
  depends on the series' length.
- **`fit_ridge` reads the raw panel.** The adstock runs from the panel's first period, so spend
  before the window carries into it; the channels are held at or above 0, a control within its own
  bounds; a missing KPI week is neither fitted nor scored. The fit hands its channels to the
  decisions as `chc.response` channels.
- **Its own trend and season, not prophet's.** A line through the window, bent at knots spaced
  evenly inside it, and harmonics of a season counted from the window, their coefficients free and
  penalised as every other.
- **`search_ridge` chooses by prediction.** Every channel's adstock and Hill and the penalty are
  searched over intervals: a scrambled Sobol sample scored, its best points polished by L-BFGS-B on
  the score's logarithm, the best refitted to the whole window. The score is the validation NRMSE
  pooled over blocks that close the window, each predicted by the model fitted to the periods
  before it; DECOMP.RSSD can be added at a weight. A fold's design reads spend alone, so a fold
  reads no KPI after its training periods.
- **Geo tests enter the fit** (`chc.lift.LiftTest`, `tests=` on both). A test's gap between its
  groups is the tested channel's coefficient times the gap in its saturated spend, under the
  channel's own transforms and the window's `K`, plus noise. So each tested channel's tests
  estimate its coefficient by least squares, and the fit is the joint least squares of the KPI and
  the tests, each period weighted by its source's noise variance: the KPI's from the fit without
  tests, the ridge's effective degrees of freedom `sum d^2 / (d^2 + n lam)` taken off, the tests'
  about their estimates, one taken off for each tested channel. The tests' rows enter
  `ridge_path` as a quadratic `Anchor` on their coefficients, the same objective less a constant.
- **The tests enter the search's score as well.** A test's gap rises and falls as the channel's
  adstock and Hill say it should, so the tests read the transforms, the shape of the curve a plan
  is made on, which a coefficient does not carry. The score is the joint least squares of the
  folds' errors and the tests' errors under the whole window's calibrated fit, each source over a
  noise variance fixed for every candidate: the KPI's the validation mean square error of the
  search without tests, run first; the tests' their observed gaps' mean square over the readout
  periods before their groups' spend parts, where no transform puts a gap. In the KPI's units, it
  is the validation NRMSE when there are no tests. Each fold's fit is calibrated as the whole
  window's is, the KPI's noise read off the fold's own training periods, so a fold reads no KPI
  after them.

## Consequences

- The oracles, each kept here as JSON under `tests/fixtures/oracles/`, its header naming what
  recorded it:
  glmnet 5.1's path, free, sign-bounded and capped, to `1e-8` of the KPI's spread; Robyn 3.12.1's
  transforms to `1e-15`; three of Robyn's models on its demo data, reached from the raw panel, the
  objective no worse than glmnet's run to `1e-14`. Robyn's own glmnet stops at its default
  threshold of `1e-7`, on the nearly unpenalised model `4.4e-3` of the KPI's spread short.
- In a world without noise the search finds the truth to `1e-5`; with noise its choice predicts the
  folds better than the truth at every penalty tried.
- A calibrated fit is the joint least squares of an independent solver in the columns' own units to
  `1e-10`; its weight is the two noise variances computed directly, the KPI's degrees of freedom
  read off the ridge's hat matrix; tests of little noise hold their channels at the truth that the
  ridge shrinks; and a calibrated fold is `fit_ridge`'s calibrated fit with the KPI from its block
  on blanked out.
- Mutations: 16 of `ridge_path`, 18 of the transforms, 20 and 23 of the model, 10 of the trend and
  season, and 25 of the search, of which 24 fail a test (polishing the worst starts reaches the
  same optimum in the test worlds); each of 41 mutations of the calibration fails one.
- A fit's tests read the coefficient alone, through the transforms its spec fixes; the transforms
  move only through the search's score.
- A calibrated search runs the search without tests first, for the KPI's noise: about twice the
  evaluations. It refuses tests with no readout period before their groups' spend parts, since
  their noise would have nothing to be read off.
- On Track M v2's scored worlds, pre-registered from a pilot of other seeds, the library's search
  plans worse than CHC's and PyMC-Marketing's plans; the pilot's ties with them did not hold. In
  regret per euro of the quarter's plan, the library's search less each arm, paired 95 %:
  - 200 drawn worlds: it beats the tests in each fold's fit alone, -0.053 [-0.077, -0.029],
    Meridian, -0.052 [-0.069, -0.035], Robyn on the 100 it planned, -0.171 [-0.219, -0.123], and
    the search without tests; it loses to CHC, +0.018 [+0.003, +0.033], and PyMC-Marketing,
    +0.020 [+0.003, +0.037].
  - 100 reference worlds: it beats the tests in each fold's fit alone, -0.082 [-0.135, -0.029],
    and the search without tests, ties Meridian, -0.012 [-0.043, +0.019], and loses to CHC,
    +0.047 [+0.017, +0.077], and PyMC-Marketing, +0.063 [+0.033, +0.093].

  `evidence/bench/results/track_m2_ridge_score.json` records the run, at mixdyn `39a0a72` and
  CHC `0ab2cd7`; `evidence/bench/track_m2_ridge.py` ran it.

## Alternatives considered

- **Robyn's search and choice.** Not reproduced: in-sample NRMSE only falls as the penalty does,
  and a choice from the Pareto front's clusters is a judgement no score states.
- **Robyn's calibration**, a third objective, the mean absolute percentage error between a model's
  implied lift and the tests', ranked with the other two. Rejected: it moves the choice of
  hyperparameters, not the coefficients, and a lift test's noise enters no weight.
- **The tests in each fold's fit alone, the score the KPI's.** The simpler; rejected on Track M
  v2's 40 pilot worlds under a rule stated before they ran: the joint score only if its paired 95 %
  interval of regret per euro against this lay below 0. It did: -0.066 [-0.107, -0.024], lower on
  20 worlds, tied on 10, higher on 10. Anchored folds keep the transforms the KPI chooses, and with
  them curves the tests contradict. On the same worlds, in regret per euro of the quarter's plan:

  | Plan | Mean [95 %] | Median |
  |---|---|---|
  | the search without tests | 0.231 [0.158, 0.303] | 0.199 |
  | the tests in each fold's fit alone | 0.107 [0.063, 0.151] | 0.059 |
  | the joint score, as the pilot wrote it | 0.041 [0.023, 0.059] | 0.026 |
  | the library's search, each fold's KPI noise its own | 0.043 [0.025, 0.061] | 0.026 |
  | the status quo | 0.150 [0.118, 0.182] | — |

  The search without tests planned worse than the status quo. The library's search tied CHC's and
  PyMC-Marketing's plans on the same worlds, +0.001 [-0.006, +0.008] and +0.004 [-0.013, +0.020].
  `evidence/bench/track_m2_ridge.py` ran it, and `evidence/bench/results/track_m2_ridge_pilot.json`
  records it, at mixdyn `ff9c8b1` and CHC `0ab2cd7`.
- **The tests' noise read off each candidate's own residuals.** Rejected: a candidate whose
  transforms misread the tests would raise their noise, and so lower their weight in its own score.
- **The KPI's noise read once off the whole window** for every fold's anchor, as the pilot read it.
  Rejected: a fold would read the KPI after its training periods through the tests' weight.
- **prophet's trend and season.** Rejected as a dependency; the model's own terms recover a bent
  trend and a season exactly in a world without noise.
