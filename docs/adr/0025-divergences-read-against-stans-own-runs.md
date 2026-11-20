# 0025. The posterior test reads divergences against Stan's own runs, world by world

Status: accepted, 2026-10-04

## Context

`test_the_posterior_is_stan_s` (`tests/test_bayes_geo.py`) asks of the engine's divergent
transitions that they be no more frequent than Stan's: the engine's share of the two samplers'
divergences is not above its share of their draws, by a one-sided binomial test at 0.001 against
the oracle's 40 000 draws (ADR 0013). The binomial test reads every transition as an independent
trial. Divergences are not independent: a chain that reaches the posterior's narrow region diverges
there several times over.

CI's run of the commit that gives chain 0 a stream of its own (ADR 0024) diverged 15 times on the
`tanh` world, against the oracle's 12 in 40 000 draws, and failed at p = 1.4e-9. The same test at
the same seed passed on the author's machine: a runner whose floating-point kernels differ moves
the chains' last bits, and with them the run. The fresh runs recorded for ADR 0024, each four
chains of 1 000 draws after 1 000 of warm-up, counted these divergent transitions per run:

| world | CmdStan 2.38, 12 seeds | the engine, 8 seeds | the oracle, 40 000 draws | the binomial test passes up to |
|---|---|---|---|---|
| normal | 0 or 1 | 0 or 1 | 3 | 4 |
| Student-t | 0; once 2 | 0; once 2 | 0 | 2 |
| `tanh` | 0 to 3; once 27 | 0 to 4 | 12 | 6 |

Stan's run of 27 fails the test the engine is held to, at p = 1e-19. On the `tanh` world a test
that reads one run against the binomial passes or fails by the run its seed and its runner draw.

## Decision

- **A run passes on divergences when the binomial test passes, as before, or when it diverges no
  more often than the most of Stan's twelve runs on that world**: 1 on the normal world, 2 on the
  Student-t world and 27 on the `tanh` world.
- The moments' bound, R-hat's ceilings (ADR 0024) and E-BFMI's floor do not change.

## Consequences

- A sampler that diverges as Stan does passes on every world, and one that diverges more than the
  binomial allows and more than any of Stan's runs still fails.
- On the `tanh` world a run of up to 27 divergences passes. What so many do to the posterior is
  caught where it matters: the moments' bound and E-BFMI still hold the run.
- ADR 0013's divergence check is now this ADR's.

## Alternatives

- **The binomial test with Stan's fresh runs pooled into the oracle's count.** Rejected: it keeps
  the independence the counts refute, and Stan's run of 27 still fails it.
- **A negative binomial fitted to Stan's twelve counts, its 0.999 quantile the ceiling.** Rejected:
  twelve counts, eleven of them 3 or fewer, fix its tail poorly, and the largest count observed
  says as much without a model.
- **A seed that passes on every runner.** Rejected for ADR 0024's reason: no seed's run is stable
  across runners' kernels.
