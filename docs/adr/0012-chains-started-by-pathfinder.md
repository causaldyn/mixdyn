# 0012. Chains started by Pathfinder

Status: accepted, 2026-10-02; amended 2026-10-03: `PathfinderStart()` is `fit_bayes`' default,
as a pilot stated before it ran decided (Consequences)

## Context

ADR 0009's sampler starts each chain where Stan's would, uniform in `[-2, 2]` on the sampling
scale. Fitted without geo tests, the Bayesian model's Hill fits of the bench's pilot worlds often
diverge, and in the worst one chain of four never mixes: its warm-up adapted the step size to a
ridge where the intercept is negative and the first channel's early-saturating curve stands in for
the baseline, and it stayed there for all 1000 draws, every parameter's R-hat above 1.08. The
density has no second hill there: L-BFGS from each of the four starts, the trapped chain's among
them, climbs to one mode. The trap is warm-up's, and a chain that starts where the posterior has
its mass does not have to cross that ridge to reach it. Stan's oracle starts its chains the same
way as the engine, so it would not catch this.

## Decision

- **`mixdyn.inference.pathfinder`**, the engine's own Pathfinder (Zhang, Carpenter, Gelman and
  Vehtari 2022), as the module's docstring states it: one path is SciPy's L-BFGS-B climbing the
  log density, 5 corrections deep, and at each iterate the normal at `x - H g` with `H` the BFGS
  inverse Hessian in Byrd, Nocedal and Schnabel's compact form over a diagonal updated as Gilbert
  and Lemaréchal update it; 25 draws estimate each normal's evidence lower bound, and the path's
  draws come from the normal whose estimate is the largest. Several paths' draws are pooled,
  weighted by the target's density over their normal's, the weights Pareto-smoothed (ADR 0011's
  PSIS), and as many as asked are taken without replacement.
- **`fit_bayes(start=...)`**: `UniformStart()`, ADR 0009's, or `PathfinderStart()`: four paths,
  each from a start uniform in `[-2, 2]`, their draws resampled into one start for each chain, as
  Stan's multi-path Pathfinder initialises chains. The four starts are four distinct draws, so no
  two chains start at one point.
- **`PathfinderStart()` is the default** of `fit_bayes` and of the nested model's stages, which
  `fit_bayes` draws. The model over geos keeps uniform starts until a pilot on its own model
  decides; the pilot below read the one-geo model only.
- Where it parts from Stan's (`single.hpp`, `multi.hpp`): the optimiser's line search and first
  step are SciPy's, so no path is Stan's; a step whose curvature fails the test is dropped from the
  history, as the paper's Algorithm 1 drops it, where Stan keeps it and skips only the diagonal's
  update. A Wolfe line search gives every accepted step a positive curvature, so the two differ
  only on a step whose `|y|^2 / (s . y)` exceeds `1e12`.
- ADR 0011's PSIS gains one case: a tail whose first quartile is the next largest ratio, where
  Zhang and Stephens' grid has no scale, is left as it is, its shape infinite, as R's `loo` leaves
  it. Resampling a path whose normal is the target exactly gives such a tail.

## Consequences

- The compact inverse Hessian is the BFGS recursion's to `1e-12`; the diagonal is the inverse of
  the diagonal of the BFGS update of the Hessian it stands for, to `1e-12`; each draw's log density
  is SciPy's normal's. An isotropic normal is found exactly from one step: at every iterate the
  normal is the target, its draws' log densities the target's to `1e-12`. A draw where the log
  density is not a number carries minus infinity and is never resampled.
- Against CmdStan 2.38's Pathfinder, one path from each of 100 uniform starts on each of two normal
  targets (`evidence/stan_pathfinder.py`, its record `evidence/results/stan_pathfinder.json`),
  scored by the divergence of the normal fitted to a path's 1000 draws from the target: on the unit
  tests' 5-dimensional normal, median 0.093 against Stan's 0.108, 90th percentile 0.26 against 0.27;
  on a 20-dimensional normal of condition number 1000, median 11.8 against 16.7. Five corrections do
  not hold twenty dimensions for either.
- On the oracle's three worlds, from three seeds each, every parameter of every chain's start lies
  within 2.75 posterior standard deviations of Stan's posterior mean, in the panel's units, where
  uniform starts lie 2 000 to 6 000 away; from Pathfinder's starts the normal world's posterior is
  Stan's within Monte Carlo error, as from uniform starts.
- Every path on the oracle's worlds runs to L-BFGS-B's cap of 1000 iterations: the log density
  keeps climbing by about `1e-3` an iteration, no coordinate running away, and Stan's tolerances
  would not stop it either. Starting four chains by Pathfinder costs four such paths and their
  evidence lower bounds; it is measured on a quiet machine before it is quoted.
- Mutations: each of the module's 28, and the PSIS guard's one, fails a test.
- *The pilot*, its rule stated in its script's docstring before it ran
  (`evidence/bench/track_m2_starts.py`, its record
  `evidence/bench/results/track_m2_starts_pilot.json`): the bench's 60 pilot worlds, each fitted
  without its tests by the Hill model from each start; Pathfinder's starts become the default if
  fewer of their fits have an R-hat above 1.01 and their divergent transitions are not more. From
  uniform starts 8 fits had such an R-hat, the largest 1.59, and 1 905 transitions diverged in 43
  fits; from Pathfinder's, 6, the largest 1.023, and 449 in 44. The plans at the posterior medians
  barely moved. Three fits in four still diverge from either start: the model's geometry, which no
  start changes.

## Alternatives considered

- **Start every chain at the mode, jittered.** Rejected: chains that start together make R-hat
  blind to a chain that would not have got there, and the mode of a posterior with scale
  parameters is not where its mass is.
- **A longer warm-up.** Rejected: the trapped chain's warm-up adapted its step size to the ridge;
  more of the same adaptation keeps it there.
- **Stan's resampling, 1000 draws with replacement, the chains' starts drawn from those.** The same
  in distribution for four starts among 4000 draws, one step longer, and two chains may share a
  start.
- **Stan's Pathfinder, run by CmdStan.** The engine's dependency policy (ADR 0001); it is the
  oracle instead.
- **Fix each channel's Hill slope at 1, as Meridian 2.1.0 does by default.** A change of model, not
  of the sampler; it removes the early-saturating curve the trap rides on, and is weighed apart.
