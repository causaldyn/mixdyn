# 0022. A kept spec's signed chains are drawn again until they mix

Status: accepted, 2026-10-04

## Context

A DLM choice (`choose_dlm`, and the media search of ADR 0020) fits every spec its
likelihood-ratio test keeps. With an effect held to a sign, `fit_dlm` draws the posterior truncated
to the signs from four Gibbs chains of 1 000 after 1 000 of warm-up (`chc.dlm.constrained_sample`),
and CHC marks draws whose chains have not mixed, R-hat above 1.01 or fewer than 100 effective draws
a chain over every state at every step and the variance, as not to be read. The choice kept them
all the same, and its readings and intervals read them.

The scorecard's first smoke run (ADR 0021) searched two worlds of each of six drift laws on its
small grid: 23 to 55 specs kept a world, and on 9 of the 12 worlds 1 to 11 of them had chains that
had not mixed. On one of those worlds, 46 specs kept and 8 not mixed:

- the quantities that mix slowest are the ones a planner reads: 40 of the 46 had their worst R-hat
  at the last week's media coefficients or the variance;
- the margin is thin: the median spec had 610 effective draws against the 400 asked, the lowest
  342, and 19 of the 46 fewer than 600;
- each of the 8 mixed when drawn again at 2 000 a chain, and at 4 000;
- the signs bind: of 20 000 draws of the posterior without them, the share whose media coefficients
  hold their signs at every week was below 0.01 for 38 of the 46 specs and 0 for 22, and the last
  week's `meta` coefficient fell below 0 in up to 99.7 % of them. A sampler that rejects the draws
  breaking a sign would draw the truncated posterior exactly, without chains, but not here.

## Decision

- **A kept spec whose signed chains have not mixed is drawn again at twice the draws a chain, at
  most twice** (`mixdyn.models.dlm_choice._kept_fit`, `_REDRAWS`), and the choice keeps the last
  draw. A fit whose chains still have not mixed says so (`ConstrainedDraws.mixed`), as before; the
  scorecard's DLM arm counts such specs among its flags.
- `fit_dlm`'s contract stands: four chains of `draws` each, whatever their mixing. The redraw is the
  choice's, where the kept specs' draws become a planner's readings.

## Consequences

- On the world above, the redraws run each of the 8 specs again at twice the length: about a third
  more sampling for that world. A world whose specs all mix costs nothing more.
- A spec drawn at 4 000 a chain holds four times the draws in memory: four chains of draws of every
  state at every step, about 60 MB a spec at 1 000 a chain on a weekly history of three years.
- The redraw's doubling, its cap and its sparing of a spec that mixed are each tested on a small
  world. That the two searches fit their kept specs through it is read, not tested: no test world
  leaves a kept spec unmixed at 1 000 a chain.

## Alternatives considered

- **2 000 draws a chain for every spec.** Rejected: twice the sampling on every world, where only
  the specs that have not mixed need it.
- **Leaving out the specs whose chains have not mixed.** Rejected: the mixture over the kept specs
  would lean toward the specs whose signs bind least, the ones that mix easily.
- **Rejection from the posterior without the signs.** Exact, and with no chains to mix, where the
  signs rarely bind; here they bind for nearly every spec kept.
- **Fewer draws, sized to the dozen readings a spec gives the planner.** Rejected: the readings are
  read only from chains that mixed, and 400 effective draws already take about 4 000.
