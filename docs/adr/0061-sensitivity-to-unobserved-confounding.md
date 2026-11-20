# 0061. Sensitivity to unobserved confounding

Status: accepted, 2026-11-01; adds `mixdyn.inference.sensitivity`, read beside ADR 0008's ridge
engine and ADR 0010's Bayesian model, and changes neither; amended 2026-11-06: measured on
Track M v2's world, and the reporting line written as `mixdyn.api.sensitivity_line`; amended
2026-11-07: the factsheet states the line, and the width on Track M v2's exogenous worlds read at
the world's truth

## Context

Every return the engines report is identified only if nothing the model leaves out moves both a
channel's spend and the KPI. A media plan bought against demand breaks that: the buyer raises
spend when sales are expected to rise, and the model credits the channel with the demand. No test
of the panel can show that this did not happen. What a report can state is how strong such a
confounder would have to be to change the conclusion, and what the return is at a strength the
reader names.

Two forms fit the two kinds of engine.

- The ridge engine's coefficient comes from a least-squares problem. Cinelli and Hazlett (2020,
  *Making sense of sensitivity: extending omitted variable bias*, JRSS-B 82(1)) state the omitted
  variable's bias on a least-squares coefficient in two partial R^2 values, the confounder's
  share of the treatment's residual variance and of the outcome's, and nothing else.
- A Bayesian posterior has no t-statistic. The marginal sensitivity model (Tan 2006, JASA 101)
  bounds the confounder's pull on the treatment's density by a factor `Gamma`. CHC's
  `ChannelSensitivity` (0.22.0) holds the sharp share of a lever's residual variance such a
  confounder can explain, `(1 - 1/Gamma)^2`, and the bias factor that share gives,
  `kappa(Gamma) = (Gamma - 1) / sqrt(2 Gamma - 1)`. Sharpness is Dorn and Guo's (2023, JASA 118)
  result for the binary model; for a lever's share CHC's docstring gives the two-valued lever that
  reaches it.

## Decision

- **`ridge_sensitivity(fit, channel, level=0.9) -> RidgeSensitivity`.** It runs on the least
  squares of the fitted periods' KPI on the fit's own design, at penalty 0. The design keeps the
  intercept and every column that varies, and each channel enters through its saturated spend.
  With `t` the channel's t-statistic and `df` the residual degrees of freedom, `f = |t| / sqrt(df)`
  and `f_q = q f`:

      partial_r2 = R2(Y ~ D | X) = t^2 / (t^2 + df)
      RV_q       = (sqrt(f_q^4 + 4 f_q^2) - f_q^2) / 2

  `RV_{q, alpha}` uses `f* = t_{df-1, 1 - alpha/2} / sqrt(df - 1)`. It is 0 where `f_q <= f*`.
  Where `f* < f_q < 1/f*` it is the root above at `f_q - f*`. Where `f_q >= 1/f*` it is
  `(f_q^2 - f*^2) / (1 + f_q^2)`, because inside the box of side `r` the outcome share that moves
  the interval most is `r / (r + f*^2) < r`. The report keeps `RV_1`, `RV_{1/2}` and both at
  `alpha = 1 - level`. `bound(r2_outcome, r2_treatment)` returns the bias,
  `se sqrt(df) sqrt(R2yu R2du / (1 - R2du))`, the estimate moved toward 0 by that bias, the
  adjusted standard error `se sqrt((1 - R2yu)/(1 - R2du)) sqrt(df / (df - 1))`, and the interval
  on `df - 1` degrees of freedom. `benchmarks` holds each control's partial R^2 with the channel
  and with the KPI. `benchmark(control, kd, ky)` returns the bound for a confounder `kd` and `ky`
  times as strong as that control (their Section 6.1).
- **`msm_sensitivity(posterior, channel, gammas, level=0.9) -> MsmSensitivity`.** At each draw
  `s` it rebuilds the draw's saturated spend for every channel and the draw's mean of the KPI.
  It then forms `sigma_s`, the root mean square of the KPI less that mean over the fitted
  periods, and `sd_s(D | X)`, the root mean square of the channel's saturated spend less its
  least squares on the draw's other channels, the controls, the calendar and the intercept. The
  confounder's allowance on the return is

      allowance_s(Gamma) = kappa(Gamma) * sigma_s / sd_s(D | X) * roi_s / coefficient_s

  `kappa` is `chc.ChannelSensitivity.factor`. The bound at `Gamma` runs from the `(1 - level)/2`
  quantile of `roi_s - allowance_s` to the `(1 + level)/2` quantile of `roi_s + allowance_s`.
  `allowance` reports the posterior median of `allowance_s` at each `Gamma`. `Gamma*` is the
  least `Gamma` at which the lower end reaches 0. It is found by bisection on `kappa`, along which
  the lower end falls, and read back by `chc.ChannelSensitivity.level_at`.
- **Each draw is rebuilt from the posterior, not from the panel.** A control's column is its
  contribution over its coefficient's posterior mean, which is exact because a contribution is
  linear in the coefficient. The calendar comes from the spec. The adstock and the curve are
  rebuilt from the draws as the model writes them. A test checks that the draws' means average
  to the fit's `estimated` within `1e-12`.
- **What CHC exposes, and what mixdyn adds.** CHC has `ChannelSensitivity.factor` and `level_at`,
  which are used as they are. It also has `msm_worst_case_mean` and `negative_control_gamma`,
  which bound a mean outcome, not a regression coefficient, so they are not used. CHC was not
  edited. mixdyn adds the step from a posterior draw to `sigma_s / sd_s(D | X)`, which is the
  least-squares geometry of `ChannelSensitivity`'s `confounding * nu(x)` for a static model's
  coefficient.
- **The defaults** are `level = 0.9`, the engine's usual interval for returns, and
  `gammas = (1, 1.1, 1.5, 2, 3)`. The paper's worked example is reproduced at `level = 0.95`.

## What each number means, and what it does not

- **A bound, not a correction.** No form estimates the confounder. Each states what the return
  would be if a confounder of a stated strength existed. A return robust to `Gamma = 3` is not
  known to be unconfounded; it is known to survive a confounder up to that strength.
- **`RV_1 = 0.14`** means that a confounder explaining 14 % of both the channel's and the KPI's
  residual variance, given the design, brings the least-squares coefficient to 0. A weaker one on
  both does not. It says nothing about a confounder strong on one side and weak on the other;
  `bound` and the benchmarks cover that.
- **The ridge's statistics are the unpenalised regression's.** The ridge coefficient is shrunk
  from it, and a geo test's anchor pulls it. Neither has a t-statistic, and `coefficient` is
  reported beside `estimate`. The return on the window's spend is the coefficient times a
  constant of the fit, so a share `q` of the coefficient is a share `q` of the return.
- **The MSM bound puts no limit on the KPI's side.** `R2yu` is 1, so the allowance is the most
  any confounder of level `Gamma` on the spend can do. It is conservative wherever the confounder
  explains less of the KPI.
- **The MSM bound acts through the coefficient at the draw's own adstock and curve.** A
  confounder that also bends the posterior's decay or saturation moves the return in a way the
  bound does not count. On a world whose curve the posterior identified poorly (log-normal spend
  that never went dark), the posterior missed the return without any confounder, so the bound
  could not cover it either. The tests use flighted spend, where the posterior holds the return.
- **`Gamma*` is a level on the spend's density, not a share of the KPI.** For a spend whose log
  is normal, the level reads as `1 / (1 - rho)`, where `rho` is the spend's maximal correlation
  with the confounder (CHC's reading of a Gaussian lever).

## Consequences

- *Tests* (`tests/test_sensitivity.py`):
  - **The paper's example.** The t-statistics come from the public Darfur survey. The test
    reproduces their Table 2 (R2 2.2 %, `RV_1` 13.9 %, `RV_{1, 0.05}` 7.6 %) and their bounds at
    1, 2 and 3 times `female`, each within a unit of the last printed digit. The 3x interval's
    lower end reads -0.006253 here and is printed -0.0062.
  - **The closed forms.** At equal shares of `RV_q` the estimate moves by `q` of itself. Both
    branches of `RV_{q, alpha}` equal the least box found by search, within `1e-6`.
  - **The in-sample identity.** On a ridge world with a hidden AR(1) driver, adding the driver to
    the regression moves the channel's coefficient by exactly the bias `bound` gives at the
    driver's own partial R^2 values, within `1e-9`.
  - **Exogenous world.** On a two-channel world in the Bayesian model's own form, seed 0, the
    allowance at `Gamma = 1.5` is below a tenth of the return and `Gamma*` is above 3.
  - **Confounded world.** An AR(1) driver pushes tv's log spend by 0.6 of itself and the KPI by
    25, which is `Gamma = 2.45`. Over 20 seeds the bound at that level holds the return in at
    least 18, and in more seeds than the posterior's own interval does.
  - `validation/sensitivity.mac` checks each closed form in exact arithmetic (Maxima 5.46).
- **Proposed reporting line for `api/documents.py`.** Another change owns that module. Text:

  > *Sensitivity to an unmeasured confounder.* {channel}'s return would reach 0 only under a
  > confounder that explains {RV_1:.0%} of both its residual spend and the KPI's
  > ({RV_{1,alpha}:.0%} to bring its {level:.0%} interval to 0); one as strong as {control}
  > would move it to {benchmark.estimate}. For the Bayesian fit: the return's {level:.0%}
  > interval stays above 0 up to a confounding level of Gamma = {Gamma*:.1f}; at Gamma = 1.5 it is
  > [{lower}, {upper}]. These are bounds under a stated strength, not corrections.

  Both result types hold only floats, strings, tuples of dataclasses and arrays, so the documents
  module can encode them as they are.

## Amended 2026-11-06: Track M v2's world, and the reporting line

- **The world.** Heusch's reference instance (arXiv:2608.21130, Table II), the bench's Track M v2
  (`causaldyn_bench.endogenous_mmm`, `EndogenousMediaMix()`'s defaults): budget feedback,
  anticipatory spend, television's bursts and paid shopping's bidding rule over an unobserved
  sentiment, true returns near 4.20, 2.90 and 1.56. Seeds 0 to 9, each fitted as the scorecard's
  Bayesian arm fits an export with its tanh family (the promotion held at or above 0, the price
  free, a trend of 3 bends, 3 harmonics of the year), on 2 chains of 500 draws after 500 of
  warm-up where the arm runs 4 of 1000; no divergence, the largest R-hat 1.03. The exogenous
  worlds are the same generator with the four mechanisms off. `scripts/study_endogenous_spend.py`
  (`sensitivity`) is the run; Monte Carlo errors are over the 10 seeds.

  | world | channel | true return | posterior median, median over seeds | `Gamma*`, median [min, max] | bound at `Gamma*` holds it | posterior's 90 % interval holds it | least `Gamma` whose bound holds it | allowance at 1.5 over the return |
  |---|---|---|---|---|---|---|---|---|
  | reference | pla | 4.195 | 10.40 | 1.229 [1.009, 1.339] | 10 of 10 | 3 of 10 | 1.073 [1.000, 1.138] | 2.2 to 3.3 |
  | reference | meta | 2.899 | 1.15 | 1.001 [1.001, 1.023] | 10 of 10 | 10 of 10 | 1 | 6.4 to 9.0 |
  | reference | tv | 1.558 | 0.73 | 1.002 [1.001, 1.003] | 10 of 10 | 10 of 10 | 1 | 7.0 to 11.1 |
  | exogenous | pla | 4.308 | 1.32 | 1.001 [1.000, 1.001] | 10 of 10 | 10 of 10 | 1 | 7.8 to 10.5 |
  | exogenous | meta | 2.948 | 1.22 | 1.001 [1.000, 1.001] | 10 of 10 | 10 of 10 | 1 | 15.8 to 22.7 |
  | exogenous | tv | 1.663 | 1.22 | 1.000 [1.000, 1.000] | 10 of 10 | 10 of 10 | 1 | 126.7 to 178.2 |
  | the tests' flighted world | tv | 1.397 | 1.38 | 226 [152, 348] | 10 of 10 | 10 of 10 | 1 | 0.026 to 0.040 |

- **The bound at the reported `Gamma*` holds every channel's true return** at all 10 seeds of the
  reference instance, 10 of 10 for each channel (a 95 % Clopper-Pearson interval of 0.69 to 1 for
  each). Paid shopping's posterior median is 2.5 times its return, Heusch's companion paper's
  observational bias (10.66 against 4.20, arXiv:2608.21128), and its own 90 % interval holds the
  return at 3 seeds of 10; the least `Gamma` whose bound holds the return is below `Gamma*` at
  every seed. Meta's and television's intervals reach 0 at a `Gamma` within 0.03 of 1, so the
  bound holds them at little cost to the claim, and says their returns are not robust to any
  confounder.
- **On exogenous worlds the bound is not narrow; the criterion's "width below 10 % of the
  return" is not met there.** Read as the allowance at `Gamma = 1.5`, as the exogenous test reads
  it, it is 7.8 to 178 times the return on the exogenous Track M v2 worlds, and 0.026 to 0.040 of
  it on the tests' flighted world over 10 seeds. The allowance is `kappa sigma_s / sd_s(D | X)`
  times the return over the coefficient: Track M v2's spend never goes dark and moves by 0.1 on
  the log beside the season the model holds, so the channel's saturated spend left after the
  other columns is a small part of the KPI's noise, and with `R2yu = 1` any confounder of that
  spend could carry the whole of it. The width is set by how little the spend moved, confounded or
  not; the world was not tuned. On those worlds the posterior also missed the return with no
  confounder (paid shopping 1.32 against 4.31), as the section above warns for spend that never
  went dark.
- **The reporting line** is `mixdyn.api.sensitivity_line(ridge, bayes, *, control, gamma=1.5)`,
  each part written where its result is given. Three changes from the proposed text: `Gamma*` to
  two decimals, since 1.01 to 1.02 at one decimal reads as 1, the level at which the interval
  holds 0 with no confounder; the robustness values to a tenth of a percent; and the benchmark
  stated as the coefficient it moves, the estimate `RidgeSensitivity` holds, of which the return
  moves by the same share.
- *Test*: `test_the_bound_at_gamma_star_holds_track_m2_s_returns`, at seeds 0 and 1 of the
  fixture `tests/fixtures/track_m2_worlds.json` (ADR 0074), so it runs without the bench: every
  `Gamma*` below 1.5 and the bound at it holding each true return.

## Amended 2026-11-07: the factsheet's line, and the width on exogenous worlds read at the truth

- **The factsheet states the line.** `mixdyn.api.factsheet(..., sensitivity=...)` takes results
  of `msm_sensitivity`, at most one for each channel the model reads, and writes each channel's
  `sensitivity_line` at `Gamma = 1.5`, with its `Gamma*`, among the engine's limitations: right
  after its line on the confounding no column holds, in the model's order. A result of a channel
  the model does not read, a second of one channel, or one not read at 1.5 is refused. Without
  results the factsheet is what it was, so `FACTSHEET` stays at version 1. *Not done:* a field of
  its own. ADR 0072 makes an added field a new version of the schema, and a document's union
  reads only dataclasses, so the field would have needed a new class and version 2 for a line
  this ADR states as text; the limitation the line quantifies was already on the page. The
  numbers stay in `MsmSensitivity`, which the file format saves (ADR 0062). The decision page does
  not change: it states a plan, and the returns it rests on are the factsheet's.
- **Why the width on Track M v2's exogenous worlds is not below a tenth of the return.** The
  allowance over the return is `kappa(Gamma) / f`, `f = beta sd(D | X) / sigma`: the channel's
  contribution left after the model's other columns, over the KPI's variation those columns and
  the channels leave. Below 0.1 at `Gamma = 1.5` needs `f > 10 kappa(1.5) = 3.54`, a channel whose
  partial R^2 with the KPI passes 0.93. `scripts/study_endogenous_spend.py width` reads `f` at
  each world's true curves and coefficients, with no posterior, at seeds 0 to 9 (median [min,
  max]; `sigma` and the parts in thousands of euros a week):

  | world | channel | sd of log spend given `X` | `beta sd(D \| X)` | allowance at 1.5 over the return | the same, `sigma` the white noise alone |
  |---|---|---|---|---|---|
  | reference | pla | 0.31 [0.30, 0.32] | 52.9 [49.1, 55.9] | 2.56 [2.09, 2.96] | 0.25 [0.23, 0.28] |
  | reference | meta | 0.30 [0.29, 0.31] | 22.1 [20.3, 24.1] | 6.0 [5.2, 7.0] | 0.59 [0.53, 0.69] |
  | reference | tv | 0.64 [0.61, 0.66] | 29.9 [28.0, 31.3] | 4.5 [3.8, 5.1] | 0.43 [0.41, 0.51] |
  | exogenous | pla | 0.094 [0.087, 0.101] | 19.0 [16.9, 20.0] | 6.9 [6.0, 8.4] | 0.68 [0.65, 0.82] |
  | exogenous | meta | 0.097 [0.088, 0.102] | 7.3 [6.5, 7.9] | 18.2 [15.3, 21.1] | 1.8 [1.6, 2.1] |
  | exogenous | tv | 0.096 [0.086, 0.103] | 0.98 [0.86, 1.19] | 134 [118, 160] | 13.6 [10.7, 16.8] |

  On the exogenous worlds each channel's log spend moves by its plan's noise alone, 0.1. The
  KPI's unexplained variation, `sigma` 364 [324, 450], is mostly the promotional uplift, which a
  0/1 reading of 70 % recall under one coefficient cannot hold (359 after the least squares), then
  the season, quality and price (87), the sentiment (67) and the white noise (38). With the true
  uplift itself a control, `sigma` falls to 113 and the allowance to 2.1, 5.9 and 44 times the
  return; with the white noise the only unexplained part, which no reading of this KPI goes below,
  to 0.68, 1.8 and 13.6.
- **Neither the estimator nor the width's definition is at fault.** Refitted at this head as above
  (seeds 0 to 9, 2 chains of 500 after 500; every draw equal to the 11-06 study's), the posterior's
  `sigma_s` is 1.04 to 1.05 times the truth's, and its allowance over the true return is 0.98 to 1.39
  times the reading at the truth. The excess comes from a curve the posterior did not identify:
  each channel's decay and half-saturation sit at their priors' medians (about 0.2, and the
  channel's mean spend), and its return near its prior's (medians over the seeds 1.32, 1.22 and 1.22 against 4.31, 2.95 and 1.66).
  Bounding the KPI's side instead, Cinelli and Hazlett's `R2yu < 1`, scales the allowance by
  `sqrt(R2yu)`: it reaches a tenth of paid shopping's return only at `R2yu` below 3e-4. And the
  posterior's own 90 % interval, at `Gamma = 1` with no allowance, is already 1.06 to 3.22 times the
  true return.
- **The criterion is not met on Track M v2's exogenous variant, no reading of that world meets
  it, and it stands as written.** There the spend moves too little beside the KPI's unexplained
  variation for the history to inform any return, confounded or not. It is met on the tests'
  flighted world (0.026 to 0.040, above), whose spend goes dark in 30 % of its weeks.
- **The reference instance re-checked at this head.** Its 10 seeds refitted as above give every
  draw of the 11-06 study again, so the bound at `Gamma*` holds each channel's true return at 10 of
  10 seeds (a 95 % Clopper-Pearson interval of 0.69 to 1), as the table above states.
- *Test*: `tests/test_factsheet.py::test_the_factsheet_states_each_channel_s_sensitivity_to_a_confounder`.
