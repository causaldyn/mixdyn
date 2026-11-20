# 0065. The convergence diagnostics, and where their thresholds come from

Status: accepted, 2026-11-06

## Context

Every sampled fit (`fit_bayes`, `fit_bayes_geo`, `fit_nested`) reports each parameter's R-hat and
effective sample sizes and each chain's E-BFMI, computed by `mixdyn.inference.diagnostics`.
ADR 0009 records in one bullet that they are CmdStan's `stansummary`'s, and ADR 0060 records the
health card the same module holds. What each number computes, where it is not a number, and which
thresholds anything compares it with, and from where, was recorded nowhere: the thresholds live in
tests, in the scorecard and in causal-hybrid-control, each stated beside its own use.

## Decision

- **What `diagnose(draws)` computes**, on `(chains, draws)` or `(chains, draws, quantities)`,
  each quantity on its own, as Vehtari, Gelman, Simpson, Carpenter and Bürkner (2021) define it
  and CmdStan 2.38's `stansummary` computes it:
  - *Split chains.* Each chain is cut in two halves, its middle draw dropped when the draws are
    odd in number, and the halves are read as `m` chains of `n` draws.
  - *Rank normalisation.* Each draw is replaced by the normal quantile of its pooled rank,
    `Phi^-1((r - 3/8) / (S + 1/4))`, `S` the split draws' number, ties given their average rank.
  - *R-hat* is the larger of the bulk R-hat, of the rank-normalised draws, and the tail R-hat, of
    the rank-normalised distances of the draws from their pooled median. Each is
    `sqrt(((n - 1) / n W + B / n) / W)`, `W` the mean of the split chains' variances and `B / n`
    the variance of their means. The tail R-hat catches chains that differ in scale, which the
    bulk R-hat does not.
  - *Bulk ESS* is the effective sample size of the rank-normalised draws: the chains' combined
    autocorrelations, each chain's autocovariance computed by FFT, summed by Geyer's initial
    monotone sequence, its last even lag kept as a bias term that lowers the variance for
    antithetic chains, and capped at `S log10 S`.
  - *Tail ESS* is the smaller of the effective sample sizes of two indicators, the draws at or
    below their pooled 5 % quantile and at or above their 95 %, each quantile linear in the order
    statistics.
  - The draws are scaled by a power of two to a range near 1 before an effective sample size is
    read: that changes only the exponents, and keeps squared deviations of draws of range `1e-200`
    or `1e200` inside the floats.
- **Where there is no number.** A quantity whose draws are one value has no R-hat or effective
  sample size: each is `nan`, and a fit's largest R-hat leaves it out (ADR 0060). Split chains of
  fewer than four draws, under eight a chain, have no effective sample size, as CmdStan has it:
  Geyer's sequence has too few lags to read. Fewer than four draws a chain are refused.
- **One difference from `stansummary`.** Where a tail's indicator is one value throughout a split
  chain, `stansummary` reads that indicator's effective sample size as half the draws' number
  whatever the chains; this module reads it as any other series', far below that on chains that
  disagree in location or scale (ADR 0009).
- **`e_bfmi(energy)`** is each chain's energy Bayesian fraction of missing information
  (Betancourt 2016): the sum of the squared changes of the Hamiltonian between transitions over
  the sum of its squared deviations from the chain's mean. Independent draws read 2; a chain that
  moves the energy slowly reads near 0.
- **No fit acts on these numbers.** A fit records them and returns; it neither refuses, warns nor
  draws again on an R-hat, an effective sample size or an E-BFMI. The thresholds that read them,
  each where it is stated and from where it comes:
  - *E-BFMI 0.3.* The module's docstring states that below 0.3 the momentum resampling explores
    the energy too slowly, and the Stan-oracle posterior tests hold every chain above it
    (`tests/test_bayes_mmm.py`, `tests/test_bayes_geo.py`, `tests/test_nested.py`; ADRs 0013,
    0024). The code cites Betancourt (2016) for the statistic and no source for the number.
  - *R-hat 1.01, or 1.1 on the Student-t world.* The posterior tests' ceiling, each world's set
    from Stan's own twelve runs on it (ADR 0024).
  - *R-hat 1.01 and 27 divergences in 4 000 draws.* The scorecard's report reads a plan whose fit
    is past either twice (`scorecard.report.RHAT` and `DIVERGED`, ADR 0035): the first is ADR
    0024's ceiling on the normal and `tanh` worlds, the second the most of Stan's own runs on the
    `tanh` world (ADR 0025).
  - *R-hat at most 1.01 and a bulk ESS of at least 100 a chain.* The DLM's draws of a posterior
    truncated to signs are read only above these (`chc.dlm.ConstrainedDraws.mixed`, citing
    Vehtari et al. 2021): computed by causal-hybrid-control, not by this module; a kept spec that
    misses them is drawn again (ADR 0022).
  - *The health card's thresholds* are the caller's; Meridian's model reviewer's are its
    docstring's example (ADR 0060).
- **Divergences and tree depth are the sampler's.** A transition diverges where its energy rises
  by more than 1000 (`MAX_ENERGY_ERROR`, CmdStan's) or overflows, and a trajectory doubles at
  most `max_depth` times, 10 by default (ADR 0009): the fits count each chain's divergent
  transitions. This module reads neither.

## Consequences

- A fit's diagnostics are CmdStan's to rounding on the cases the estimators treat apart, and a
  difference from a Stan run's numbers is the sampler's, not the summary's.
- A caller that wants a fit refused on its diagnostics applies a threshold itself; the engine
  ships none as a default, and the numbers above are each a test's or a report's, with its own
  source.
- `diagnose`, `e_bfmi` and the health card's functions are reached by the module's path: no
  `__all__` lists them, so under ADR 0062 they are not yet public names.
- The tests that hold the definitions, in `tests/test_diagnostics.py`:
  - `test_the_diagnostics_are_stansummary_s`: R-hat to `1e-12` and bulk ESS to `1e-10` of CmdStan
    2.38's `stansummary` on seven sets of draws (`tests/fixtures/oracles/stan_diagnostics.json`),
    and the tail ESS as `stansummary` reads it but for the one difference above;
  - `test_the_cases_cover_what_the_estimators_treat_apart`: chains autocorrelated negatively at
    the `S log10 S` cap, ties, an odd number of draws and a constant tail indicator, among the
    seven;
  - `test_chains_that_disagree_have_few_effective_draws_in_their_tails`,
    `test_several_quantities_are_each_their_own`, `test_a_constant_quantity_has_no_diagnostics`,
    `test_split_chains_of_fewer_than_four_draws_have_no_effective_sample_size`,
    `test_the_effective_sample_size_reads_no_scale` (bit for bit under a power of two, at scales
    `2^-900` to `2^900`), `test_too_few_draws_are_refused` and
    `test_e_bfmi_is_the_squared_steps_over_the_variance` (a random walk below 0.01, independent
    draws at 2 within 5 %).

## Alternatives considered

- **ArviZ's or CmdStan's summaries as a dependency.** Not taken: ADR 0001 keeps them out of the
  runtime, and `stansummary` is the oracle the tests read instead.
- **Thresholds as defaults on the fits**, refusing or warning on an R-hat above 1.01. Not done:
  the posterior tests' own record (ADR 0024) shows Stan's chains crossing 1.01 on the Student-t
  world about one run in ten, so a fixed ceiling would refuse fits Stan's own runs make, and the
  ceilings in use are each set for its world.
- **`stansummary`'s half-the-draws reading of a constant tail indicator.** Not copied: it reads
  chains that disagree as well mixed in their tails, where this module's reading falls with them.
