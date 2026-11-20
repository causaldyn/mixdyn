# 0049. A geo test's design randomised within matched pairs

Status: accepted, 2026-10-28; adds to ADR 0030's candidate tests the design of a geo test, the
geos it treats, and its power before it runs; amended 2026-10-28 by ADR 0055: `power` reads by
default with `read_did` on the groups' sums, `log_did` by `read=`, and the numbers quoted below
move: the four-week minimum detectable lift 0.040 to 0.043, by readouts of 2, 4 and 8 weeks 0.056,
0.044, 0.035, by pre-periods of 18, 26 and 52 weeks 0.099, 0.053, 0.044 (13 is refused), and the
size over 2000 independent markets 0.051; amended 2026-11-06: `stratified_design`, the comparator
design written from Kerman, Wang and Vaver's paper, ships opt-in; on the bench's families 8 and 10
the pairs read by `read_did` detect 0.73 times its lift and stay the default

## Context

`mixdyn.experiments` values candidate lift tests (ADRs 0030, 0045) but does not say which geos a
test treats, how alike its two groups are, or how likely it is to read a lift of a given size.
Three ways of choosing the treated geos fail in ways a test can be checked for:

- **Treating the top geos** by the KPI maximises the imbalance. Over ten geos whose pre-period
  means run 1 to 10, treating the top five has a squared standardised difference of 10 on the
  matching variable (Maxima), `sqrt(10) = 3.16` pooled standard deviations.
- **A greedy control match**, each treated geo in the data's order taking its nearest unused
  control, can take a control a later geo needed more: with treated geos at 0 and 1 and
  candidates at 0.9 and -2 it matches with a summed gap of 3.9 where 2.1 is possible.
- **A power formula over the geos' spread across the market** reads the error of a difference in
  levels between groups, which neither the test's length nor its pre-period moves. On twenty
  geos of log-normal levels (log-SD 0.5) and 5 % weekly noise, ten treated, such a formula read a
  minimum detectable lift of 0.34 for a four-week test; a simulation of a pre/post log
  difference in differences read about 0.035.

## Decision

- **`pair_geos(score, size=2)`** sorts the geos by a pre-period score and cuts the order into
  blocks of adjacent geos. On one dimension that is the least summed range over every partition
  into blocks; where the count does not divide, the geos left over are those whose leaving out
  keeps that sum least (dynamic programming over the sorted order), and `Blocks.left_out` names
  them. On a tie the later geo in the order is left out.
- **`assign(blocks, seed, share=0.5)`** treats a fixed share of each block, one geo of a pair, by a
  draw from the seed's own stream: matched pairs randomised within pairs (Imai, King and Nall
  2009, Statistical Science 24(1)). Every geo is treated with its block's probability, so the
  difference in means is unbiased over the draws; the pairs bound the imbalance a single draw can
  have. Over the ten geos above the 32 designs average a difference in means of 0 and their worst
  squared standardised difference is 1/10 (Maxima), all five of one side treated.
- **`match_controls(treated, candidates, features)`** serves a treated set the user fixes: the
  one-to-one match minimising the summed Mahalanobis distance by the Hungarian method
  (`scipy.optimize.linear_sum_assignment`), under the two groups' pooled within-group covariance.
  A singular covariance is refused, naming the features it cannot tell apart.
- **`balance(design, features)`** reads each feature's standardised difference, the difference in
  means over `sqrt((s_t^2 + s_c^2) / 2)`, and the largest in absolute value.
- **`power(kpi_history, design, readout, lifts, *, pre, reps, seed, alpha=0.05, read=log_did)`**
  simulates from the KPI's own history. A replication draws a window of `pre + readout` periods,
  multiplies the treated geos' KPI over the readout by `1 + lift` and reads it. `log_did` reads the
  treated less the control geos' mean log KPI a period, its mean over the readout less its mean
  over the pre-period, with the standard error of the same statistic over the placebo windows of
  the readout's length inside the pre-period, widened to a t's quantile at their effective count
  as ADR 0045 widens its own. The minimum detectable lift is where the power, interpolated from
  the test's size at 0, reaches 80 %. The windows are drawn by one uniform number a replication,
  shared by the lifts and by calls at one seed. `read=` takes any reader of the same signature.
- On the twenty-geo world above, at a 52-week pre-period of a 156-week history, the four-week
  test's minimum detectable lift is 0.040 over 2000 replications. Over 600 replications it falls
  from 0.054 to 0.038 to 0.025 over readouts of 2, 4 and 8 weeks, and from 0.052 to 0.042 to 0.038
  over pre-periods of 13, 26 and 52. Over 2000 independent markets, one window each, the test
  rejects a lift of 0 at 0.055, within three binomial standard errors of 0.05. The minimum
  detectable lift sits above the 0.035 of the simulation in the context because its standard
  error is read from the pre-period's placebo windows and widened for their few, not known.

## The comparator design (amended 2026-11-06)

The field's reference design for a geo test is GeoX's: the geos stratified on their pre-period
series' mean, coefficient of variation and slope, candidate designs drawn within the strata, kept
where a time-based regression of the treated sum on the control sum fits out of sample and passes
an A/A test, and ranked by that regression's minimum detectable lift (Kerman, Wang and Vaver 2017,
"Estimating ad effectiveness using geo experiments in a time-based regression framework"). Running
GeoX's code is a third-party oracle, which mixdyn does not take; `stratified_design(kpi_history,
readout, *, pre, seed, candidates=200, size=2, share=0.5)` is written from the published method
alone (the module states each step), and `StratifiedDesign` holds its design, its strata, the
candidates drawn and kept, and the chosen one's R^2 and formula MDE. It refuses a pre-period under
three times the readout, as ADR 0055's audit does, and a history where no candidate passes.

`scripts/study_geo_design.py` compares the two on the bench's families 8 (ten geos, policies 0
and 0.5) and 10 (three, eight and fifty geos): the first 30 pilot worlds of each of the five
environments, 150 worlds of 104 weeks. Each design reads weeks 1 to 52 (the pairs on the geos'
mean KPI there, the comparator with `pre=26` and the world's seed); each is audited by ADR 0055's
`audit_power` over the 49 backtests whose readouts lie in weeks 53 to 104, a four-week test read
after 26 weeks, at lifts 0.001 to 1. The comparator refused 4 of the 30 three-geo worlds, where
both of its two candidates failed the A/A test: 146 worlds and 7154 backtests a reader for it,
150 and 7350 for the pairs. Each design is read by both readers, so the design's part and the
reader's can be told apart. The median MDE, and the geometric mean ratio of the shipped design's
(pairs, `read_did`) to the comparator's (strata, `read_tbr`) and of the two designs' under
`read_did`, with 95 % t intervals over the worlds:

| environment | worlds | pairs, did | strata, tbr | pairs, tbr | strata, did | shipped / comparator | pairs / strata, did |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all | 150 (4 refused) | 0.0632 | 0.0892 | 0.0917 | 0.0602 | 0.729 [0.682, 0.779] | 1.056 [0.992, 1.124] |
| policy-0 | 30 | 0.0552 | 0.0666 | 0.0773 | 0.0536 | 0.895 [0.784, 1.021] | 1.091 [0.947, 1.258] |
| policy-0.5 | 30 | 0.0589 | 0.0777 | 0.0702 | 0.0487 | 0.755 [0.662, 0.861] | 1.100 [0.946, 1.280] |
| geos-3 | 30 (4 refused) | 0.1522 | 0.1967 | 0.2627 | 0.1213 | 0.669 [0.556, 0.804] | 1.118 [0.966, 1.295] |
| geos-8 | 30 | 0.0577 | 0.0812 | 0.0771 | 0.0627 | 0.757 [0.649, 0.883] | 0.968 [0.822, 1.140] |
| geos-50 | 30 | 0.0709 | 0.1167 | 0.1207 | 0.0714 | 0.596 [0.520, 0.682] | 1.018 [0.899, 1.152] |

The mean size over the 49 backtests, the mean share of rejections over the 13 whose readouts do
not overlap, the share of audits flagged by ADR 0055's binomial test over those 13, and the median
MDE range:

| environment | pairs, did | strata, tbr | pairs, tbr | strata, did |
| --- | --- | --- | --- | --- |
| all | 0.050, 0.054, 0.000, 0.053-0.076 | 0.045, 0.043, 0.000, 0.062-0.119 | 0.035, 0.039, 0.000, 0.072-0.116 | 0.062, 0.065, 0.000, 0.049-0.069 |
| policy-0 | 0.063, 0.062, 0.000, 0.045-0.062 | 0.054, 0.054, 0.000, 0.048-0.083 | 0.033, 0.031, 0.000, 0.066-0.087 | 0.078, 0.072, 0.000, 0.039-0.059 |
| policy-0.5 | 0.050, 0.054, 0.000, 0.047-0.064 | 0.048, 0.049, 0.000, 0.049-0.089 | 0.029, 0.026, 0.000, 0.056-0.091 | 0.053, 0.067, 0.000, 0.037-0.053 |
| geos-3 | 0.046, 0.059, 0.000, 0.104-0.171 | 0.031, 0.033, 0.000, 0.146-0.272 | 0.043, 0.049, 0.000, 0.189-0.306 | 0.045, 0.050, 0.000, 0.096-0.139 |
| geos-8 | 0.044, 0.051, 0.000, 0.045-0.065 | 0.039, 0.036, 0.000, 0.061-0.097 | 0.034, 0.036, 0.000, 0.068-0.089 | 0.069, 0.069, 0.000, 0.048-0.070 |
| geos-50 | 0.049, 0.046, 0.000, 0.054-0.086 | 0.054, 0.044, 0.000, 0.076-0.163 | 0.035, 0.054, 0.000, 0.094-0.158 | 0.063, 0.067, 0.000, 0.054-0.081 |

No audit is flagged: none of the 592 reads 4 rejections of its 13 (ADR 0055). The band first
written, [0.03, 0.08] on the size over all 49, had flagged 0.50 to 0.73 of each cell's audits.

- **The shipped design stays the default.** It detects 0.73 of the comparator's lift over all
  the worlds and at most 0.895 in each environment; only on policy 0 does its interval,
  [0.78, 1.02], reach 1. The better design is the one already shipped; the comparator ships
  opt-in, for a reader who wants GeoX's design and a regression read of it.
- **The gap is the reader's, not the design's.** Under `read_did` the two designs are within 6 %
  (1.056, interval [0.992, 1.124]); under `read_tbr` the pairs read 0.0917 and the strata 0.0892.
  `read_tbr`'s placebo error is the wider (ADR 0058 found it conservative on null panels), and
  its mean size over the backtests, 0.035 to 0.045, is the lower.
- **The strata read by `read_did`**, the least median MDE, 0.0602, is not adopted: its ratio to
  the pairs' holds 1, and its mean size, 0.062, is the highest of the four arms, 0.078 on
  policy 0.

## Consequences

- The power is conditional on the history: a short history holds few windows, overlapping ones,
  and the replications re-draw them, so the power is read to no better than the windows' count.
- `log_did` weighs each geo alike, as a regression of the log KPI on geo and period effects does,
  so with a lift that differs between geos it reads the geometric mean lift, not the lift of the
  groups' summed KPI. Summing the groups first, which reads the latter, gave a minimum detectable
  lift 5 % to 20 % higher over six draws of the world above, the larger geos' noise weighing
  more.
- The placebo windows are each read against the rest of the pre-period, `pre - readout` periods,
  where the test is read against all `pre`: their error is slightly the larger, so the test errs
  on the side of its size.
- The standard error is widened at the 95 % quantile, whatever `alpha` the caller tests at.

## Alternatives

- **GeoX's stratified design as the default** (amended 2026-11-06): not taken; on families 8
  and 10 it detects a lift 1.37 times the pairs' (the inverse of 0.729), and it refuses where no
  candidate passes its A/A test, 4 of 30 three-geo worlds. It ships as `stratified_design`.
- **Complete randomisation** of half the geos: unbiased, but a single draw can be as unbalanced as
  the top-N design; the pairs remove that tail at no cost to unbiasedness.
- **Top-N selection** and **a greedy match**: rejected above.
- **A cross-sectional power formula**: rejected above; it cannot move with the test's length.
- **A synthetic control for one to three treated geos** (Abadie and Zhao 2021, "Synthetic
  Controls for Experimental Design", arXiv 2108.02196): chooses the treated geos and their
  weighted controls together, so that both the treated and the control units fit the market's
  average, and reads the effect with a permutation inference. It serves a test too small to
  randomise within pairs. Left for later: it needs a constrained quadratic program over the
  geos' subsets and its own inference, and the pairs serve the tests of ten or more geos the
  candidates price now.
