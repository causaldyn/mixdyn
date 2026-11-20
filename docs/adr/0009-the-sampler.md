# 0009. The sampler: our own NUTS, Stan's where Stan can be replayed

Status: accepted, 2026-10-02

## Context

The Bayesian engine draws from a posterior over a few dozen coordinates: each channel's return,
carryover and curve, the base, the noise. Through the transforms that free them of their
constraints the posterior is curved, and its scales differ by orders of magnitude. The No-U-Turn
sampler (Hoffman and Gelman 2014) is the standard for such a posterior; Meridian and
PyMC-Marketing both sample with it, and Stan's is the reference implementation. ADR 0001 rules out
Stan, NumPyro, BlackJAX and PyMC as dependencies, so the sampler is written here, and needs an
oracle before anything uses it.

## Decision

- **NUTS as CmdStan 2.38 runs it.** Each transition draws a momentum and doubles a leapfrog
  trajectory in a random direction until the generalised no-U-turn criterion fails, checked around
  each merged pair of subtrees and across the pair. Its sample is drawn in proportion to each
  point's `exp(-H)`, uniformly within a subtree and biased towards a new subtree between it and the
  trajectory (Betancourt 2017, appendix A). A step whose energy rises by more than 1000, or
  overflows, diverges. The acceptance statistic is the mean over every leapfrog step, a discarded
  subtree's included. The metric is diagonal.
- **The reference first, in NumPy float64, by recursion**, one transition at a time. It is the
  implementation a faster one is checked against.
- **The transition compiled** (`mixdyn.inference.compiled`) for a density written in JAX, which
  the engine's models fit with. Its subtrees are built by iteration: the leaves in time order,
  each completed subtree merged with its left neighbour as the leaves' binary count carries, so the
  merges come in the reference's order and read the same uniforms. Warm-up, the step size searches
  and the random numbers stay the reference's, in NumPy, so the two samplers share everything but
  the transition.
- **A transition's random numbers are drawn before it** (`Noise`): a normal for each coordinate of
  the momentum and a uniform for each decision it can make. Two implementations given the same
  `Noise` make the same decisions, so a compiled or vectorised sampler can be compared with this
  one transition by transition, not only in distribution.
- **Warm-up is Stan's**, kept apart from the transitions in `Adaptation`, so that it can be told a
  warm-up Stan ran:
  - the step size by dual averaging towards a mean acceptance of 0.8;
  - a diagonal metric over windows that double from 25 transitions, after an initial 75 and before
    a terminal 50, the last stretched to the terminal buffer;
  - each window's variance shrunk towards `1e-3` by `5 / (n + 5)`, in Stan's order of operations,
    and a window of one position shrinking the metric it had, as Stan does;
  - the step size searched for at the start, from 1 under a unit metric, and where a window closes,
    under its metric, then dual averaging restarted towards ten times the search's result;
  - after warm-up, the exponentiated average of the log step sizes since the last search.
- **One difference from CmdStan 2.38.** Below 150 transitions of warm-up and from 20, Stan shrinks
  the buffers to 15 % and 10 % of warm-up, as its manual says, but returns before it restarts the
  windows, so the window between the buffers never closes and no metric is adapted (CmdStan's
  `windowed_adaptation`, and its runs: 100 and 149 transitions leave a unit metric, 150 and 200
  adapt it). This module closes that window.
- **The diagnostics are `stansummary`'s** (Vehtari et al. 2021): split R-hat, the larger of the
  rank-normalised draws' and of their folded distances from the median; bulk and tail effective
  sample sizes by Geyer's initial monotone sequence; and each chain's E-BFMI (Betancourt 2016).
  One difference: where a tail's indicator is one value throughout a split chain, `stansummary`
  reads that indicator's effective sample size as half the draws' number, and this module as any
  other series'.

## Consequences

- The oracles, each recorded from CmdStan 2.38 and kept here as JSON under
  `tests/fixtures/oracles/`:
  - *Exactness:* 3000 exact draws of a Gaussian and of a curved target, each moved two or three
    transitions, are still exact draws (Kolmogorov–Smirnov per coordinate).
  - *The kernel is Stan's:* 20 000 transitions at the step size and metric Stan ran, from its
    start, distribute their tree depths as Stan's 200 000 (a chi-square test), and every statistic,
    the leapfrog steps, the acceptance, the divergences, each coordinate's second moment and the
    lag-1 autocorrelations of it and of its square, lies within 4.5 joint standard errors of
    Stan's. A valid sampler that is not
    Stan's fails: one that samples between the trajectory and a new subtree uniformly, or checks no
    turn across merged subtrees.
  - *Warm-up is Stan's to the bit:* told the acceptance statistics and positions of four of
    CmdStan's warm-ups, of 150 to 500 transitions, one to four windows, one of them a window of one
    position, `Adaptation` gives every transition the step size Stan gave it and leaves Stan's step
    size and inverse metric, each compared with `==`. Each search starts a power of 2 from Stan's
    result, the first under a unit metric, the last under the metric warm-up leaves.
  - *The search is Stan's in distribution:* its first step sizes fall on the powers of 2 as
    CmdStan's did over 4000 chains of each of two targets (a chi-square test).
  - *The diagnostics are `stansummary`'s* to `1e-10` on seven sets of draws, each a case the
    estimators treat apart: chains autocorrelated positively and negatively, the second at the cap
    on the effective size, ties, an odd number of draws, heavy tails, and chains that disagree in
    location and in scale.
- The replay found two defects of the first version, both at the windows' end: a shrink one unit in
  the last place from Stan's, computed in another order, and a window of one position, whose
  variance was not a number.
- Mutations: of 61 of the reference sampler, its warm-up and the diagnostics, 59 fail a test, and
  the other two cannot change a result: clamping an acceptance statistic that never exceeds 1, and
  a factor of 1. Each of 26 of the compiled transition fails one.
- A warm-up shorter than 150 transitions gives a different metric from CmdStan's, by design; the
  replay oracle is recorded at 150 and above.
- The compiled transition makes the reference's decisions: from 900 random points of three
  targets, at step sizes from far too small to far too large, the same depths, leapfrog steps and
  divergences, and the same sample, acceptance and energy to `1e-12`. Whole chains are compared in
  distribution only: a chain is sensitive to rounding, a start moved by `1e-15` ending 150
  transitions of warm-up `1e-6` away, so two implementations' chains part after a few hundred
  transitions however alike their arithmetic.

## Alternatives considered

- **Stan, NumPyro or BlackJAX as a dependency.** Ruled out by ADR 0001.
- **Static HMC.** Rejected: a trajectory length to tune for each model, which NUTS chooses for each
  transition.
- **A compiled NUTS first.** Rejected: an iterative tree in a compiled loop has more places to
  differ from Stan than a recursive one, and nothing to be checked against transition by transition.
- **CmdStan's short warm-up, kept as it is.** Rejected: the manual and the warning Stan prints both
  say the stages are rescaled to 15 %, 75 % and 10 % of warm-up, and a warm-up that adapts no
  metric samples a posterior whose scales differ by orders of magnitude with one step size.
- **A dense metric.** Not now: Stan's default is diagonal, and the oracles are recorded for it.
