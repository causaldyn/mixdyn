# 0101. A third confirmatory run draws from a region of its own

Status: proposed, 2026-11-23; adds `Declaration.third_confirmatory`, read on a bench whose
`causaldyn_bench.scorecard.family.confirmatory` takes `third`; a declaration that does not set it
is written as before.

## Context

ADR 0098 split each confirmatory block of 1 000 seeds between two runs: the first run's 300 and
the second's 700. The second run (M4) read up to 520 worlds an environment, so a block holds at
most 180 worlds no run has read, fewer than a family at the cap of 520 needs.

The first two runs also read family 17's two environments, the twins `curve` and `line`, from one
shared block: the bench draws a pair of twins as one seed read in both environments. The claim
pools a family's environments and reads every world as independent, so on family 17 it counted 240
worlds where 120 seeds were drawn. Read with the seed as the cluster, every verdict of M4 on
family 17 still held, but the pooled reading overstated the worlds.

## Decision

- **The bench draws a third run from a region of its own.** Family `f`'s `e`-th environment, `e`
  counted in the order the family names its environments and family 0 included, draws from
  `1 000 000 + 20 000 f + 1 000 e`, a block of 1 000 seeds in order. The region lies past every
  family's block, every confirmatory block, every seed a committed run drew and every range
  development runs drew from; the bench's tests hold it apart from all of them, and the tests'
  noise of its worlds, `1000 s + c`, apart from every committed seed.
- **Each environment has a block of its own, the twins too.** The third run reads `curve` and
  `line` from different seeds, so the claim's pooled reading counts independent worlds. The twins'
  point, a history that cannot tell the two truths apart, holds in distribution: both environments
  draw their histories from one generator.
- **The run declares which.** `third_confirmatory` is refused on a run that is not confirmatory or
  that is also the second. It is one of the fields a pool's runs declare alike (`AGREED`), and a
  joined root and a root beside a run must match it.

## Consequences

- A third confirmatory run reads up to 1 000 worlds an environment.
- A first or second confirmatory run calls the bench as before.
- Family 17's verdicts on a third run read twice the seeds of a second run at the same declared
  worlds.

## Alternatives considered

- **The rest of each second block, seeds 820 to 999.** Rejected: 180 worlds an environment is below
  the cap, and the first family to need more would need a fourth rule.
- **Keeping the twins on one seed, read with the seed as the cluster.** Rejected for this run: the
  claim's reader reads independent worlds, and a clustered reader is a change to every verdict's
  arithmetic, not to one family's seeds.
