# 0081. The claim by family and rung, and a scored run as the next run's development set

Status: proposed, 2026-11-16; amends ADR 0040 and ADR 0080: a run may declare its claim read by
family and rung, one intersection-union test a rung, the verdicts tested at 0.025 in a fixed
sequence the run declares, its comparisons predicted to fail read and left out, Hall's word printed
where Hall's reader reads the look; a run that does not declare it reads as before, line for line

## Context

The claim of ADR 0040, as M2's pre-registration states it (`scorecard/preregistration/m2.md`),
is one intersection-union test a family over every comparison at both rungs. The next run reads
the engine again on fresh worlds, and five things it needs the code did not do:

- **A verdict a rung.** The engine's arm at k = 0 and its arm at the tested rung are different
  procedures, and the next run selects each apart. One test over both rungs fails a family on
  either rung and says nothing of the other.
- **A comparison known to fail before the run.** A pre-registration that predicts a comparison
  fails can still read it, but the family's test then fails by construction. The next run claims
  a comparison only where its predictive, read on worlds no candidate was tuned on, is at least
  one half, and runs and prints every other one.
- **One reader a verdict.** Under Hall's reader the claim's level and its within were Hall's, but
  the printed word, `beats`, `ties` or `loses to`, was Student's interval's. On skewed differences
  the two disagree (`tests/scorecard/test_claim_by_rung.py` builds one).
- **The two-stage arm's first stage.** The rule that reads a fit past its chains' bounds
  (`scorecard.report.past`) read the two-stage arm's DLM alone, not the static arm's fits whose
  curves its first stage hands it.
- **A development set and fresh worlds.** The next run tunes its arms on the scored run's worlds,
  whose rivals' records exist, and is confirmed on worlds no scored run drew. Nothing split those
  worlds, paired a pilot's new arms with the stored records, or allocated the fresh seeds; and the
  harness kept neither a line for each attempt nor a process's cost.

## Decision

- **The claim by family and rung** (`Declaration.per_rung`, `scorecard.claim.verdicts`). At each
  rung of each family read, the comparisons in the claim are one intersection-union test (Berger
  1982), read at the largest of their levels within the margin (`Comparison.within_level`).
- **A fixed sequence over the verdicts** (`Declaration.order`, each a family and a rung). The run
  declares the order its verdicts are tested in, every verdict that claims a comparison named
  once; the pre-registration sets it from each verdict's predictive, the likeliest first. Each
  verdict is tested at 0.025 in that order, and holds where it is shown and every one before it
  held. The first that is not shown ends the sequence: every verdict after it is read and printed
  `not confirmed: the sequence stopped at family F rung R`. The order names only verdicts the
  claim reads: each root's declaration refuses an entry of its own family that the claim does not
  read of it (`Declaration.reads`), where the run is exploratory, does not hold the rung, or has
  every rival of its own there predicted to fail or capped exploratory. A verdict the order names
  that is still not read, a family whose root is absent, stops the sequence; a verdict that claims
  a comparison and that the order does not name is refused. The fixed sequence holds the
  familywise error at 0.025 with no adjustment (Maurer, Hothorn and Lehmacher 1995), since each
  verdict is tested only once all before it held.
  `claim read` prints the verdicts in the order, then the headline: an intersection-union test
  over every verdict at 0.025, said only where every family and rung is claimed whole and holds.
  Every root of a run that reads the claim declares the same order, or `claim read` refuses them.
- **Predicted failures** (`Declaration.predicted`, each a rung and a rival). Each is read and
  printed with its line marked `not claimed: predicted to fail`, and left out of its verdict; a
  verdict whose every comparison is predicted to fail is printed `not claimed`. A declaration
  predicts failures only where it declares the claim by rung.
- **Hall's word.** Under the claim by rung with Hall's reader, the printed word is Hall's reader's:
  `loses to` where it shows the rival below the default, `beats` where it shows the default below
  the rival, `ties` otherwise.
- **Secondary readings, never read by the claim** (`scorecard.claim.secondary_lines`), printed after
  the verdicts: the comparisons with every fit past its bounds at any stage at the status quo's
  regret; the comparisons at the margin 0.01; and at rung 0 of each family the best plan less the
  status quo and Meridian's plan less the status quo, each with its paired 95 % interval.
- **Every stage of a fit** (`scorecard.report.past`). A two-stage record is past its bounds where
  its DLM did not mix or a static fit its first stage read is past R-hat 1.01 or its share of
  divergent draws.
- **The development set** (`scorecard.development`). A scored run's worlds are split once: within
  each family and environment, ranked by seed, every third world from the third is `check`, and so
  is every world of a `<root>-robyn` root; the rest are `fit`. A pilot's arms on those worlds are
  joined to the stored records by family, environment, seed and rung, only where both exports hold
  the same observation by digest and the built-in arms score alike. The selection rule scores a
  candidate at a rung by its worst mean difference over the confirmatory families and rivals on the
  fit split; the smallest wins, ties within 0.005 go to the fewest CPU-seconds a fit, then to the
  fewest settings. The winner's predictive is a two-stage bootstrap on the check split: each
  environment's check worlds drawn with replacement, then the declared worlds drawn from those, each
  draw read as the claim reads it, each comparison on its complete cases by Hall's reader against
  the margin. The rule counts a candidate's failed fit as the status quo, as the bench scores it,
  and prints each candidate's failures on each family: a candidate's robustness is part of its
  quality. The predictive draws a world where either arm failed and leaves it out of that
  comparison, so it predicts what the claim, read on complete cases, will read.
- **Confirmatory worlds** (`Declaration.confirmatory`, the bench's
  `causaldyn_bench.scorecard.family.confirmatory`). A confirmatory run draws each environment's
  block of worlds that no scored run reads: the family's scored block of environment `e + 9`, and
  family 0's blocks of its own, 1 000 worlds an environment each. The bench lists the scored run's
  worlds and pilots among the seeds committed runs drew, and its test holds every confirmatory
  block apart from all of them. A run's roots, its pool's, its joined root and the roots beside
  it, are all confirmatory or none, or the claim refuses them.
- **The harness's record of its processes** (`scorecard.run`). Every attempt of a world's process
  is noted as it starts and as it ends (`logs/attempts.jsonl`), and its log gets a line naming the
  attempt that follows, appended to, never overwritten. Inside its scope each process appends its
  cost as it ends: the scope's CPU-seconds and peak memory from its cgroup, its wall and its exit
  (`logs/costs.jsonl`). An arm's limits name the CPUs a process of it takes (`Limits.cpus`), by
  default a tool's measured share (`scorecard.tools.RESERVED`), for a stage's admission to reserve.
- **Records reused** (`scorecard.run reuse`). A tool's record from another run's root is taken in
  place of a fit where the run's export of the world is the observation the record names by digest,
  a version-4 export read as the version 2 the tool was handed, and the tool's pinned files, its
  script, its lock and the bench's scripts they import (`scorecard.tools.fingerprint`), hash alike
  at both runs' commits of the bench, but the files accepted by name (`reuse --accept`): a file
  changed only where the tool's default setting does not run, an option it adds, which the
  pre-registration names with its evidence. Each record taken is noted in `logs/reused.jsonl`,
  with the accepted files that moved.
- **Nothing else moves.** A declaration without `per_rung`, `order`, `predicted`, `confirmatory`
  or CPUs other than 1 is written as one was before, so every existing run reads and is declared
  again unchanged, and `claim read` over it prints what it printed, byte for byte: M2's claim
  over every comparison, with Holm's step-down over its losses.

## Consequences

- Each verdict is tested at 0.025, so a run sizes N a verdict at 0.025, with no inflation for the
  number of verdicts. The price is the order: a verdict after the first that fails is not
  confirmed, however strong its evidence, so the order puts the verdicts likeliest to hold first,
  and the pre-registration fixes it before the run.
- The rule and the predictive read a failure apart: the rule counts a candidate's failure
  against it, and the predictive drops the world from the comparison, as the claim on complete
  cases does. A candidate that fails often loses the rule before its predictive is read. The
  predictive does not model the claim's rule that leaves out an environment where the rival failed
  on more than half of the worlds.
- A comparison predicted to fail costs its worlds and enters no verdict. The predictive decides
  what is claimed before the run, on worlds the selection did not read, so the claim's scope is
  not chosen on the confirmatory worlds.
- The headline is rarer than a family's verdict: one comparison predicted to fail anywhere keeps it
  from being said, though every verdict holds.
- The secondary readings print more lines after every reading by rung. They are labelled as never
  read by the claim.
- The pilot fits no rival on the scored run's worlds, and Robyn's stored plans serve its families.
  Reuse refuses a tool whose pinned files moved unless each moved file is accepted by name: the
  hash cannot tell an option added beside the default from a change to it, so the acceptance and
  its evidence are the pre-registration's, not the code's.
- The cost line is written by a small Python wrapper inside the scope (`COST`), which ends as its
  process ended, by the same signal or code, so a failure's recorded ending reads as before; a
  shell would have turned a signal into a code. The scope's argv is M2's, so a stage's driver that
  wraps `_scoped(limits, command)` still does. A process stopped at its timeout or by a machine's
  stop leaves no cost line; its attempt's end line says how it ended.

## Alternatives considered

- **The claim over every comparison of a family, both rungs, as before**, with predicted failures
  dropped from it. Simpler, but one rung's failure still fails the family, and the two arms are
  selected apart. Rejected.
- **Holm's step-down over the verdicts.** It needs no order and stops at no verdict, but its first
  step reads at 0.025 over the verdicts claimed: at some fourteen verdicts a verdict needs about
  1.8 times the worlds for the same power at a true tie. The predictive gives the order the
  sequence needs. Rejected.
- **The selection on the complete cases**, the worlds both arms planned. A candidate that fails
  where it would plan badly loses those worlds from its mean and looks better than it is.
  Rejected.
- **The predictive with a failed fit at the status quo.** It would predict the claim's secondary
  readout, not the claim, which reads complete cases. Rejected.
- **Holm over every comparison** rather than over verdicts. It reads each comparison at a level
  that depends on all the others and loses the intersection-union structure a verdict needs.
  Rejected.
- **A comparison claimed by its fit-split mean alone.** It ignores the mean's own error: a mean on
  the margin is claimed as often as one well inside it. The two-stage predictive carries it.
  Rejected.
- **Fresh worlds from the rest of each scored block**, past the worlds the scored run read. The
  sample would no longer be a prefix of its environment's order, and the next scored run would
  overlap it. Rejected for blocks of their own.
- **Reuse by the record's export bytes.** An archive's bytes move with the zip's timestamps while
  its observation does not. The digest is what the tool names. Rejected.
- **Reuse whatever the tool's files.** A tool's script that changed its default would hand its old
  records to a new run as if they were the new script's. Rejected; a moved file is refused unless
  accepted by name.
