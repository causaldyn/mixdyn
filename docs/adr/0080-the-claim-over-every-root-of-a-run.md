# 0080. The claim over every root of a run

Status: proposed, 2026-11-10; amends ADR 0040: `claim read` is handed every root of a run, reads a
pool whose runs test at different rungs above 0, reads the tools of the roots beside a run with the
run, and, asked with `--recorded`, prints every recorded arm less every rival after the claim; a run
handed alone, without roots beside it, reads as before; amended 2026-11-16 by ADR 0081: a run
may declare its claim read by family and rung

## Context

ADR 0040's claim reads each run's own comparisons. Since then the claim reads one family over
more than one root in two ways: a joined root, `<root>-ts`, holds the default above rung 0, and a
pool reads several runs as one run over their environments (`Declaration.joined`,
`Declaration.pool`). The M2 pre-registration (`scorecard/preregistration/m2.md`) lays its run out
in more roots than that, and states three readings the claim did not make:

- **Family 7 tests at its own rung in each environment**: 1 on `L-52`, 2 on `L-78` and `L-104`,
  so its two runs declare rungs (0, 1) and (0, 2). The pool refused runs that declare different
  rungs. The sizing reads the family as one comparison a rival at the tested rung
  (`scorecard/preregistration/m2_sizing.py`), its records above 0 at one label.
- **Tools in roots of their own.** The runner plans a capped tool on every world of its root, so
  Robyn, capped at 20 worlds an environment, runs in `<root>-robyn` beside each main root, with
  `chc`. Meridian on family 10 runs in `f10-meridian` on `geos-3` and `geos-8` alone, and
  `bayes-geo` in `f10-bayes-geo` on the same two. The pre-registration reads Robyn's comparisons on
  its 20 worlds and Meridian's on family 10's two environments, both exploratory. `claim read`
  read neither: handed a root without a margin, it stopped.
- **Readings outside the claim.** The pre-registration reports regardless every recorded arm less
  every rival, `bayes geo plug-in` on family 10 among them, and `dlm two-stage` less `stacked` at
  the tested rung. Before the freeze, the frozen commit has to compute them. `scorecard.report`
  reads each root alone, and its recorded arms against that root's own arms, so it reads neither:
  `bayes-geo`'s root holds no tool, and the two-stage arm and `stacked` are in two roots.

## Decision

- **A pool's runs may test at different rungs above 0** (`scorecard.claim.tested_rungs`). Runs of
  one pool that each hold rung 0 and one rung above it, not all the same one, may differ in their
  rungs; on everything else the claim reads they agree, as before (`AGREED`). They also run under
  one pre-registration and plan the same arms, as the roots of one name beside them plan the same
  arms (`BESIDE`). The pool is read at the first run's rungs: each run's records at its rung above
  0 are read at the first run's, so a rival's comparison above 0 is one comparison over every
  environment, read once in the intersection-union test and once in Holm's list. Its line names
  each environment's own rung: `the tested rung: k = 1 on L-52, 2 on L-78 and L-104`. Runs that
  differ in their rungs in any other way are refused as before.
- **A root beside a run** (`read_runs`, `read_beside`). Handed a root that declares no margin,
  `claim read` reads it as a run's joined root, as before, or as a root beside a run handed: a
  directory beside the run's, named by the run's directory name, a hyphen and a name of its own.
  It must run under the run's pre-registration; hold the run's family, its pilots or scored worlds,
  its bench and directory of worlds, some of its environments at every rung of the run; and plan
  an arm the run does not, on the run's worlds or on a cap of each such arm, which only a tool can
  declare. Otherwise it is refused, and so is a root without a margin beside no run handed.
- **What is joined.** Each arm the root plans and the run does not is joined to the run's records
  by environment, seed and rung (`join`), the built-in arms checked alike in both roots, as the
  joined root's arm is. An arm both plan, `chc` beside Robyn, is the run's; the root's own is not
  read.
- **What is compared.** Each tool of the root is a rival of the run's default, read as a tool of
  the run is (`compare(..., beside=...)`): on the root's environments, its line saying
  `on geos-3 and geos-8 only` where they are not all the run's; on its cap, where the root caps
  it; exploratory where the root declares its caps so. A world the root does not hold yet leaves
  the comparison unread. An environment the root does not hold is not the comparison's: its
  worlds are not paired and hold none of the tool's failures. The root's own arms, `bayes geo
  plug-in` and `bayes geo cvar 0.1`, are joined and compared with nothing in the claim, since the
  claim reads its default against rivals alone; they are reported beside it (below).
- **The roots beside a pool's runs** are read like its runs: each run of the pool is handed the
  root of the same name beside it (`f7a-robyn` and `f7b-robyn`), or the claim is refused, and the
  roots of one name, agreeing on their worlds, arms, tools and caps (`BESIDE`), are read as one
  root over every environment they hold.
- **Reported regardless** (`recorded`, `claim read --recorded`). After the claim's lines,
  `claim read --recorded` prints two readings at each rung of each run as the claim reads it, the
  pools, the joined root and the roots beside the run included. Each recorded arm, an arm of this
  engine's that the run or a root beside it plans and that is not the rung's default, less every
  rival: the built-in arms, the run's tools and the tools of the roots beside it. And at each rung
  above 0 where a run declares its default there, that default less `stacked`, the engine's static
  arm (`STATIC`). A pair is read on the complete cases of the environments both arms are read on,
  at the fewest worlds an environment either root reads, so a capped tool on its cap, by the claim's
  own `complete_case` and the bench's `paired`. Its line gives the mean, the paired 95 % Student-t
  interval, the complete cases and the reading rule's verdict, unadjusted, with each arm's
  failures. It enters neither the claim nor Holm's list.
- **Nothing else moves.** The declaration gains no field. A run handed without roots beside it
  reads as it did, line for line; `claim read` without `--recorded` prints what it printed; and
  `scorecard.report` reads each root alone, as before.

## Consequences

- M2's `claim read` is handed every root of the run. Meridian's root on family 10 declares
  Meridian capped at the family's worlds an environment with its comparisons exploratory: a root
  without a margin makes a tool exploratory only through a cap, and a cap at the root's own worlds
  reads all of them.
- Family 7's comparison at the tested rung pools rungs whose worlds hold different numbers of lift
  tests. The claim prints one interval over the three environments, not one an environment.
- The roots of a run are tied by their names. A root named for a run is read beside it, and a root
  that matches no run handed is refused, so a mistyped name stops the reading instead of dropping
  the root's comparisons.
- Every recorded arm less every rival is read twice. `scorecard.report` reads one root at a time,
  each of its arms against that root's other arms, over every world of the look, the status quo
  standing in for a failed plan, as before. `claim read --recorded` reads across the run's roots,
  on the complete cases. The pre-registration's items are the second: there `bayes geo plug-in` on
  family 10 is read against Meridian and every other rival on `geos-3` and `geos-8`, and
  `dlm two-stage`, from `<root>-ts`, less `stacked`, from the main root.
- The recorded arms are read off the declarations, not off a list. Every variant of the arms a run
  and its roots plan is read, the `vs equal` variants and `bayes geo cvar 0.1` too, which the
  pre-registration's table of arms calls recorded and its list of readings does not name. A run
  that does not plan `stacked` prints no line of the default less it.
- Pools and roots beside them that plan different arms, and a root beside a run that adds an arm
  on fewer worlds than the run's without a cap, are now refused. The reading would read such an arm
  on part of a pool's environments, or on a prefix of worlds that nothing declares. M2's roots
  declare neither.

## Alternatives considered

- **A field naming the roots beside a run**, as `joined` names the `-ts` root. It states the layout
  in the main root's declaration, and changes every main root's `run.json`. The names already tie
  the roots, and the claim refuses a root it cannot place. Rejected.
- **Every tool read on fewer environments than the run's left out of the claim by rule.** It hides
  a choice the declaration can state, and does state, through the cap. Rejected.
- **Two comparisons a rival above 0 in family 7**, one a run. It reads the family at two sizes and
  doubles its comparisons there, where the pilot sized it pooled. Rejected.
- **The readings outside the claim printed by every `claim read`.** No flag to forget, but every
  reading of the claim grows by some hundreds of lines and stops being the claim's alone. The flag
  keeps `claim read`'s output as it was, and the pre-registration names the flag. Rejected.
- **The pre-registration's list of recorded arms in the code.** It prints the eight arms the list
  names and no more, but ties the module to M2's names, and the list omits four variants its own
  table calls recorded. Rejected: the declarations name the arms, and reading more is not
  selective.
- **The default less every recorded arm**, in place of `stacked` alone: no arm's name in the code,
  but some ten readings a rung that nothing pre-registered. Rejected: `STATIC` names one arm.
