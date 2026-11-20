# 0055. A geo test's power read by the reader that reads it

Status: accepted, 2026-10-28; amends ADR 0049's `power`: its default reader is ADR 0047's
`read_did` on the groups' summed KPI, not `log_did`; amended 2026-11-06: `audit_power` reads a
design's power once over every backtest, refuses a pre-period under three times the test and fewer
than 8 backtests whose readouts do not overlap, flags its size by an exact binomial test over those
and reports an MDE range; `power` is unchanged

## Context

ADR 0049's `power` simulates a geo test from the KPI's own history and reads each replication with
`read=`, by default `log_did`: the treated less the control geos' mean log KPI, each geo weighed
alike, its standard error from every placebo window of the readout's length inside the
pre-period, each read against the rest of the pre-period. Two things are wrong with that default:

- **Its size where the noise persists.** ADR 0047 rejected placebo windows read against the rest
  of the pre-period for its readers: a window between two stretches of its baseline is read too
  tightly where the noise carries over from week to week, and a prototype rejected a true null at
  0.058 to 0.072. `power`'s own test does the same, so its power is optimistic on an
  autocorrelated KPI.
- **Its estimator.** Once it runs, the test is read by `mixdyn.experiments.reading`'s readers, on
  the treated and the control groups' summed KPI, with walk-forward placebo windows. A power
  simulated for `log_did` is the power of a test nobody reads.

mixdyn is unpublished, so the default can move.

Measured: one window of each of `n` independent markets, so the rejections are independent, of
twenty geos of log-normal levels (log-SD 0.5), half treated within pairs matched on their
pre-period mean (ADR 0049's world), a four-week readout, no lift, a 5 % test. The log noise is
independent at 5 % a week, or a stationary first-order autoregression of coefficient `rho` and
standard deviation 5 %. Each reader of `reading.py` is adapted to `power`'s `Reader` by a private
wrapper, its estimate's lift and its standard error widened to its t (so `power`'s normal test is
the reading's verdict at 5 %); `read_synthetic_control` reads the treated geos' mean against each
control geo, not the control sum. `n = 8000` (two seed blocks of 4000, each block's binomial
standard error at 0.05 is 0.0034, the pooled 0.0024), `n = 4000` for `variance="iid"`:

| noise | pre | `log_did` | `read_did` | `read_tbr` | `read_synthetic_control` | `read_tbr`, iid |
| --- | --- | --- | --- | --- | --- | --- |
| independent | 26 | 0.0473 | 0.0514 | 0.0450 | 0.0430 | 0.0505 |
| independent | 52 | 0.0406 | 0.0373 | 0.0356 | 0.0464 | 0.0503 |
| `rho = 0.5` | 26 | 0.0541 | 0.0570 | 0.0509 | 0.0456 | 0.181 |
| `rho = 0.5` | 52 | 0.0553 | 0.0423 | 0.0375 | 0.0466 | 0.172 |
| `rho = 0.8` | 26 | 0.0778 | 0.0680 | 0.0661 | 0.0359 | 0.323 |
| `rho = 0.8` | 52 | 0.0600 | 0.0429 | 0.0448 | 0.0418 | 0.301 |
| `rho = 0.95` | 26 | 0.1130 | 0.0749 | 0.0720 | 0.0365 | 0.433 |
| `rho = 0.95` | 52 | 0.0871 | 0.0539 | 0.0556 | 0.0364 | 0.430 |

Two binomial standard errors of 0.05 at `n = 8000` are 0.0049. No reader is within them on every
row. At `rho = 0.8` and a 52-week pre-period, over six seed blocks of 8000 (48 000 markets,
standard error 0.0010), `log_did` rejects 0.0564 and `read_did` 0.0415.

## Decision

- **`power`'s default reader is `read_did` on the groups' sums**, at `reading.py`'s default
  `warmup = 13`, through a private wrapper `_summed` that builds the window's panel and test and
  returns the estimate's lift and widened standard error. The simulated power is the power of
  the test as `read_did` will read it, its size included, whatever that size is.
- Of the group-sum readers, `read_did` and `read_tbr` are within 0.006 of each other on every
  row; `read_did` is the closer to 0.05 on both independent rows, the panels the existing size
  test reads, and over a 52-week pre-period it does not pass the band's top, 0.0539 at most,
  where `read_tbr` reads 0.0556 at `rho = 0.95`. Over 52 weeks it under-rejects where the noise
  is independent or `rho <= 0.8`: there the power is pessimistic, not optimistic.
- The wrapper refuses a pre-period that holds fewer than two placebo windows of the readout with
  13 periods before each (`pre >= readout + 14`), naming the counts, as `log_did` refuses fewer
  than two of its own.
- `log_did` stays public and selectable by `read=log_did`.
- Tests: over 8000 AR(1) markets at `rho = 0.8` and a 52-week pre-period, the default rejects at
  most `0.05 + 2 sqrt(0.05 0.95 / 8000)`, and `log_did` rejects more often on the same markets,
  by over three standard errors of the paired difference (six seed blocks: 4.3 to 6.7). Under the
  old default the first assertion fails, at 0.0600. The bound is one-sided: `read_did`'s size
  there, 0.0415, lies below a two-sided band.
- ADR 0049's quoted numbers move, on the same world and seeds: the four-week test's minimum
  detectable lift over 2000 replications 0.040 to 0.043; over 600, by readouts of 2, 4 and 8
  weeks, 0.054, 0.038, 0.025 to 0.056, 0.044, 0.035; by pre-periods, 0.052, 0.042, 0.038 at 13,
  26 and 52 to 0.099, 0.053, 0.044 at 18, 26 and 52 (13 holds no placebo window of 4 after 13
  periods); the size over 2000 independent markets 0.055 to 0.051.

## The power audit (amended 2026-11-06)

`power` draws its windows, so the same window can be read many times and a short history's power
is the best of few draws. A design chosen on few backtests promises more power than it delivers:
mlsynth's replication of GeoX's design reports 0.77 delivered where 16 backtests promised 0.80,
and recommends 8 backtests or more and an MDE range. `audit_power(kpi_history, design, readout,
lifts, *, pre, read=read_did)` reads each backtest once, every window of `pre` periods and the
`readout` after them one period apart, at every lift, at a 5 % level:

- It refuses a pre-period shorter than three times the readout, naming both and the length it
  needs, and a history that holds fewer than 8 backtests whose readouts do not overlap, naming
  their count and the history it needs, `pre + 8 readout` periods.
- `PowerAudit` reports the backtests' count, the power at each lift, the size (the share of
  backtests that read a lift at 0), the MDE at 80 % over all of them, and the MDE range: the
  least and the largest of the MDE read over the first half of the backtests in time and over
  the second.
- **The flag.** The size over every backtest is the estimate, reported; whether it differs from
  5 % is tested over the backtests one `readout` apart from the first, whose readouts do not
  overlap: `disjoint` of them, `rejected` of which read a lift at 0. The audit is `flagged` where
  the exact two-sided binomial test of that count against 5 %, twice the smaller tail, at most 1,
  rejects at 1 % (`flag_p_value <= 0.01`). The flag is reported, not a refusal.
  validation/geo_aa_flag.mac checks the rejection region in exact rationals for every count of
  backtests to 150: the flagged counts are an upper run, from 4 of 8, 4 of 13, 5 of 19, 6 of 25
  and 8 of 49, their chance under a size of 5 % at most 0.01 (0.00037, 0.0031, 0.0020, 0.0012 at
  8, 13, 19 and 25), and a lower run only from 104 backtests on. Below 104, no count is too low to
  flag: the flag reads a size too large, never one too small. It catches a reader far from its
  size, not one near it: over 13 backtests a reader of size 0.10 is flagged with probability
  0.034, of 0.15 with 0.118, of 0.20 with 0.253 and of 0.30 with 0.579.
- Why apart. Backtests one period apart share their readouts' periods, so their rejections are
  not independent, and a binomial test of their count flags a reader of the right size too
  often. Over 4000 null audits of four geos of independent unit noise, a 4-week test after 12
  over 64 weeks (49 backtests, 13 apart), read by a reader whose verdict rests on its readout
  alone and whose size is exactly 5 %: the binomial test over the 13 flagged 0.0023 of the audits
  (its exact chance 0.0031); the same test over all 49 flagged 0.028; and ADR 0058's band
  [0.03, 0.08] on the size over all 49, the rule first written, 0.68. A test holds the first
  within 3 standard errors of 0.0031 and the second above 0.01 by over 3. The readers' errors
  also read the pre-periods, which backtests apart still share, so for them the count is not
  exactly binomial and the flag is that much rougher.
- **The flag on ADR 0049's study**, re-read under this rule: 592 audits (146 and 150 worlds by
  two designs by two readers), each of 13 backtests apart. Their counts of rejections apart were
  0 in 279, 1 in 244, 2 in 63 and 3 in 6, against binomial chances of 0.513, 0.351, 0.111 and
  0.021; none reached the 4 that is flagged, so no audit was flagged (the least p-value 0.049),
  where the band had flagged 0.66 of the shipped design's. The arms' mean share of rejections
  apart, 0.039 to 0.065, sits beside their mean size over every backtest, 0.035 to 0.062; the
  MDEs and sizes of the re-run are the first run's, value for value.
- `read_did` and `read_tbr` are linear in the lift: it moves a backtest's effect by the lift
  times the treated sum over its readout and leaves its error alone. A test holds `audit_power`
  to that computation on 49 backtests, and the study reads its 1000 lifts off one reading a
  backtest, checked against `audit_power` on every tenth lift of each environment's first world.
- `power` does not take the three-times rule: its existing calls, a four-week test after 18
  weeks among them, read as before.

## Consequences

- A design's power is the power of the reading that will follow it: a test planned at 80 % is
  read by the same estimator, its walk-forward placebo windows and its t included.
- `read_did` over-rejects over a 26-week pre-period where the noise persists, 0.068 at
  `rho = 0.8` and 0.075 at 0.95: its ten placebo windows read too few of the noise's long swings.
  That is the reading's own size, which `power` now reports rather than hides; it belongs to
  `reading.py` and is left to a decision of its own. Over 52 weeks it holds.
- Where the noise is independent or `rho <= 0.8` over 52 weeks, `read_did` under-rejects, 0.037
  to 0.043, and the minimum detectable lift reads the higher for it; on ADR 0049's world it moves
  from 0.040 to 0.043.
- A short pre-period is refused that `log_did` read: the walk-forward windows need 13 periods
  before the first. The shortest pre-period of a four-week test is 18.
- `log_did` reads the geometric mean lift, `read_did` the lift of the summed KPI (ADR 0049); with
  a lift that differs between geos the default now reads the latter, which is what the model
  calibrates on.
- Each replication builds a `MediaPanel` and a `GeoTest` for the reader.

## Alternatives

- **Keep `log_did`**: rejected; it over-rejects at `rho >= 0.8`, 0.060 to 0.113, and is not the
  estimator that reads the test.
- **`read_synthetic_control`**: never past the band's top on any row, 0.036 to 0.047, but below
  its bottom at `rho >= 0.8` and over 26 independent weeks; it reads single control geos, not the
  control sum, and needs three of them and 8 blank periods.
- **`read_tbr`**: within 0.006 of `read_did` on every row, further from 0.05 on the independent
  ones, and past the band's top at `rho = 0.95` over 52 weeks.
- **`read_tbr(variance="iid")`**: exact where the noise is independent, 0.17 to 0.43 where it
  persists (ADR 0047's 0.32).
- **The audit's flag at ADR 0058's band [0.03, 0.08] on the size over every backtest**
  (amended 2026-11-06): the rule first written, replaced the same day. No share of 8 backtests
  lies in the band (1 of 8 is 0.125), so every audit of 8 is flagged; of 49 independent
  backtests a reader of size exactly 0.05 falls outside it with probability 0.519 (binomial,
  Maxima); over the null audits above it flagged 0.68, and over ADR 0049's study 0.50 to 0.73 of
  the audits in each environment and reader, 0.66 for the shipped design at a mean size of 0.050.
- **The binomial test over every backtest** (amended 2026-11-06): its count is not binomial,
  the backtests sharing their readouts; it flagged 0.028 of the null audits above at a level of
  0.01.
- **A reader whose size is 0.05 on every row**: none of the five is; finding one is a change to
  `reading.py`'s standard errors, not to `power`.
