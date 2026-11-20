# 0024. The posterior test holds R-hat where Stan's own chains hold it, world by world

Status: accepted, 2026-10-04

## Context

`test_the_posterior_is_stan_s` (`tests/test_bayes_geo.py`) runs the engine's four chains of 1 000
draws after 1 000 of warm-up on each of the three worlds the Stan oracle records: normal errors,
Student-t errors and a `tanh` curve. It held every parameter's R-hat below 1.01 on all three
(ADR 0013), beside the check that matters for the posterior: every mean and standard deviation
within `BOUND` joint Monte Carlo errors of Stan's 40 000 draws.

When chain 0 stopped reading the stream the starts are drawn from (seed audit, 2026-10-04), the
Student-t world's largest R-hat at the test's seed went from below 1.005 to 1.0274, on the
`search` channel's geo deviations and spread: one chain's effective draws on that coordinate fell
to 24 of 1 000. Every other check held. Whether that is the engine or the world was measured by
running both samplers again from fresh seeds, each run four chains of 1 000 after 1 000 of
warm-up on the oracle's own data and model.

The largest R-hat of each run, over every parameter:

| world | CmdStan 2.38, 12 seeds | the engine, 8 seeds |
|---|---|---|
| normal | 1.0025 to 1.0068 | 1.0029 to 1.0057 |
| Student-t | 1.0027 to 1.0683; one run above 1.01 | 1.0026 to 1.0274; one run above 1.01 |
| `tanh` | 1.0028 to 1.0099; two runs at 1.0095 or more | 1.0024 to 1.0040 |

Stan's run above 1.01 read 1.0683 on a geo spread, with two divergent transitions. On the engine's
streams before the audit the Student-t world also had one run of eight above 1.01, 1.0182. In all
32 of the engine's runs, these eight and the 24 above, every moment held within `BOUND` (the worst
3.03 to 4.33 against 6.23),
every chain's E-BFMI was at least 0.667, and its divergences passed the binomial test against
Stan's. On the Student-t world both samplers cross 1.01 about one run in ten: a test that reads
one run there passes or fails by the run its seed draws, and a runner whose floating-point kernels
differ can draw another run from the same seed.

## Decision

- **The R-hat ceiling is the world's**: 1.1 on the Student-t world, 1.01 on the normal and `tanh`
  worlds. Each is the least of 1.01, 1.05 and 1.1 above every one of Stan's twelve runs on that
  world; on the normal and `tanh` worlds every one of the engine's eight is under it too.
- The moments' bound, the E-BFMI floor of 0.3 and the divergence test do not change. They check
  the posterior; R-hat checks only that the chains mixed.

## Consequences

- A sampler that mixes as Stan does passes on every world, and chains that stall, whose R-hat runs
  far past 1.1, still fail. Slower mixing than Stan's on the Student-t world that stays under 1.1
  is not caught by R-hat; what it does to the posterior is caught by the moments' bound.
- On the `tanh` world the ceiling sits next to Stan's largest. A failure there at 1.01 is read
  against Stan's runs before it is called a defect of the engine.
- ADR 0013 holds every R-hat below 1.01; on the Student-t world the ceiling is now this ADR's.
  ADR 0018's R-hats below 1.005 were read on the streams before the audit.

## Alternatives

- **1.01 everywhere, with a seed that passes.** Rejected: it fixes one run, which another runner's
  arithmetic need not reproduce, and it hides the measurement that Stan fails the same ceiling.
- **More draws, so that R-hat's spread narrows below 1.01.** Rejected: twice the draws doubles
  the test's time for a tighter R-hat, on a check the moments' bound already makes.
- **No R-hat check.** Rejected: a stalled chain can sit inside a loose moments' bound for one run,
  and R-hat names it directly.
