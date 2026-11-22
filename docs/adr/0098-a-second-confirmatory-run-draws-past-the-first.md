# 0098. A second confirmatory run draws past the first

Status: proposed, 2026-11-20; adds `Declaration.second_confirmatory`, read on a bench whose
`causaldyn_bench.scorecard.family.confirmatory` takes `second`; a declaration that does not set it
is written as before.

Numbered 0096 on the branch where M4 was frozen: M4's pre-registration
(`scorecard/preregistration/m4.md` and `m4.json`, unchanged since the freeze) cites it as ADR 0096.

## Context

A confirmatory run reads worlds no scored run read: each environment's confirmatory block, a
prefix of its seeded order (ADR 0081). The next scored run must read worlds no earlier run read,
the first confirmatory run's included. A prefix of the same block would hand it the first run's
worlds again, and worlds read from the middle of a block would hold places the reader's looks do
not start from.

## Decision

- **The bench splits each block.** The first confirmatory run's worlds are each block's first
  `SECOND_RUN` (300), the second run's the remaining 700, each in its own seeded order: a world's
  index starts at 0 in its run's order, so the second run's looks read as the first's do. The
  first run read at most 240 worlds an environment, so its worlds and their indices are unchanged.
- **The run declares which.** `second_confirmatory` is refused on a run that is not confirmatory.
  It is one of the fields a pool's runs declare alike (`AGREED`), and a joined root and a root
  beside a run must match it, so a run's roots draw the same worlds.

## Consequences

- A first confirmatory run now reads at most 300 worlds an environment; a second one at most 700.
- A first confirmatory run calls the bench as before, so a run declared on an earlier bench still
  reads there.

## Alternatives considered

- **A block of new seeds for each run.** Rejected: every family's block is tiled by its pilot,
  scored and confirmatory blocks, and a new range would need a third order `index` resolves.
- **Skipping each block's first worlds by the count the first run declared.** Rejected: the count
  differs by family and would live in the declaration, where one constant states the split once.
