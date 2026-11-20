# 0023. A media search fits a stratified sample of its kept specs, each from a start its sampler survives

Status: accepted, 2026-10-04

## Context

A media search (`mixdyn.models.dlm_media.search_media`, ADR 0020) keeps every spec it filtered
that a likelihood-ratio test at its level does not reject, and fitted each one: with an effect
held to a sign, four chains of 1 000 after 1 000 of warm-up of the posterior truncated to the signs,
drawn again where they have not mixed (ADR 0022). The scorecard's DLM arm reads 400 readings of the
kept specs' channels, spread evenly over them.

On the smoke run's small grid the searches kept 23 to 55 specs a world. On the arm's full grid,
on one of those worlds (`trend`, seed 220101), 1 964 of the 2 491 specs the search filtered were
kept, at 11 parameters, whose test admits every spec within 9.84 of the best log-likelihood: the
data barely separate the transforms. Fitting them all would take about 9 CPU-seconds and 60 MB of
draws each, about 5 CPU-hours and 110 GiB a world; the arm's process was killed at a 10 GiB cap
after 35 minutes. And 400 readings over 1 964 specs read one draw from at most 400 of them: the
rest would be fitted and never read.

The same search, with every kept spec drawn at 10 a chain to count them, met another failure: one
kept spec's sampler stopped. CHC's exact Hamiltonian trajectory (`chc.dlm._reflected_orbit`) raises
where one trajectory meets more than 1 000 walls a constrained value, and this spec's did from
seed 0, at 10 draws a chain and at 1 000. From seeds 1, 2 and 3 its chains mixed, R-hat at most
1.005 and at least 1 085 effective draws. The stop depends on where the chains start, and it
took the whole search, and the arm's records of the world, with it.

## Decision

- **`search_media(..., fitted=K)` fits at most `K` of its kept specs**, at evenly spaced ranks of
  their log-likelihood: rank `floor(i (N - 1) / (K - 1))` of the `N` kept, for `i` from 0 to
  `K - 1`, so the best is fitted first and the last kept last. `None` fits every one, as before.
- The choice holds every kept spec and its filter (`specs`, `filtered`), and the fits of those
  fitted (`kept`). The readings, `bounds` and `coefficient_bounds` read the fitted ones: the
  equal-weight mixture over the kept specs becomes a sample of it, stratified by likelihood, each
  fitted spec standing for the `N / K` kept about its rank.
- **A kept spec whose sampler stops is drawn from the next seed**, at most four
  (`mixdyn.models.dlm_choice._drawn`, `_SEEDS`); from the last, the stop is raised. The stop is
  the only `RuntimeError` `chc.dlm` raises; any other is retried alike and raised from the last
  seed as well.
- **The scorecard's DLM arm fits 32**, and its flags record the specs kept, those fitted and how
  many of these have not mixed.

## Consequences

- A world's cost and memory no longer grow with the grid. With 64 fitted, on one world of each of
  the drift family's six laws (k = 0, measured under load, so planning figures): 11.5 to 22.7
  CPU-minutes and 5.1 to 8.8 GiB at the peak a world; every fitted spec mixed, 9, 21 and 33 of
  them after a redraw on three worlds, and no sampler stopped. Fitting 32 samples half the specs
  and holds half the draws, and memory bounds how many worlds one machine runs at once: at 64 one
  world came within 1.3 GiB of the 10 GiB cap the full grid's run was given.
- How many to fit was read off the same worlds: each world's 64 fitted specs split into disjoint
  stratified subsets of 8, 16 and 32, each subset's 400 readings planned as the arm plans.
  On five of the six worlds `dlm mixture`'s plan sat on a vertex of the box and the budget, two
  of its three channels at a bound, and every subset planned it, at the plug-in's regret. On
  `walk`, one channel at a bound, its regret per euro moved between subsets by an sd of 0.10 at 8
  and 16 specs and 0.04 at 32. `dlm cvar 0.1` moved by an sd of at most 0.05 at 8, 0.04 at 16 and
  0.02 at 32. Across the six worlds the two arms' regrets at 32 run from 0.003 to 0.69 per euro:
  at 32 a world's draw of specs adds little to the variance a comparison over worlds carries.
- The union of intervals that `bounds` and `coefficient_bounds` give is over the fitted specs, so
  narrower than the union over all the kept would be.

## Alternatives considered

- **Every kept spec.** Infeasible at the full grid, above.
- **64.** Its plans would move less between subsets, for more time and twice the draws held; at
  32 the subsets already move a world's regret far less than the worlds differ.
- **Leaving out a spec whose sampler stops.** Its next seed mixes, and a spec left out would lean
  the sample toward the specs whose sampler never stops.
- **Raising, as CHC does.** One start of one spec in 1 964 would fail the world's every DLM record.
  CHC could instead keep a trajectory that meets the cap where it started, which keeps the chain's
  law if the starts that meet it are closed under reversal; that is CHC's to show and release.
- **`K` drawn at random.** Unbiased too, but it needs a seed of its own, and a draw may leave a
  stretch of the likelihood's ranks unfitted, where the stratified ranks cover every stretch.
- **A coarser grid, or a lower level.** Either changes the arm. The kept share says the data
  barely separate the transforms, and a coarser grid would hide that rather than resolve it.
- **Weights by likelihood.** A different arm: ADR 0020's mixture ranks no kept spec above another.
