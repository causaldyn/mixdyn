# 0021. The scorecard's harness

Status: accepted, 2026-10-04; amended 2026-10-06 by ADR 0035: a record's cost holds its
compute, a failure names its kind, and a plan whose chains are past their bounds is read twice;
amended 2026-10-11 by ADR 0039: the tools' arms run by the same rule, each world a process admitted
by a memory gate, under limits the run declares; amended 2026-10-11 by ADR 0040: a run declares
the default procedure and the margin the claim reads it by, and its report ends with the claim;
amended 2026-10-11 by ADR 0041: the engine's ridge model and CHC's own planner are arms beside
`bayes` and `dlm`, neither reading a lift test; amended 2026-10-12 by ADR 0043: a family with a
geo axis is read, a national arm planning its market and the model over geos every cell; amended
2026-10-12 by ADR 0044: each arm that holds more than one reading also plans its worst tenth
against the equal split; amended 2026-11-06: a run declares the SHA-256 of its pre-registration,
its report can end with the bench's report of the family, and the bench's families 17 to 20
are in `FAMILIES`; amended 2026-11-07 by ADRs 0077 and 0078: the calibration loop, the decision
tools, the geo readers, a switchback and the geo DLM are arms; amended 2026-11-07 again: a geo
family is exported at the bench's version 4, a tool is handed it as version 2, and two variants
are scored on the truth beside the bench's score

## Context

The engine is to be compared with the maintained media-mix tools on worlds whose truth is known.
The worlds, their truth, the scoring and the sequential gate live in the public bench
(`causaldyn_bench.scorecard`): a family of worlds draws each world from a seeded stream, exports
what an arm may read of it once, with a digest over every array of the export, and scores any
arm's record of a plan against the world's best plan, beside the status quo and the equal split.
The tools' arms are the bench's scripts, each run in its own pinned environment over the archived
exports, each writing one JSON record a world.

The engine needs the same: its plans made from the same exports, recorded in the same form, scored
by the same function, so that every comparison is paired over the same worlds. The engine is not
a dependency of the bench and the bench is not one of the engine's, and the runs are long, run on
a rented machine as well as this one, and must survive a process killed for memory.

## Decision

- **The harness lives in the engine's repository, outside `src/`,** in `scorecard/`, so it ships
  with the sources and not in the wheel. The bench enters through `PYTHONPATH`, never the lock;
  `ty` checks the harness with the bench's imports allowed to stay unresolved
  (`allowed-unresolved-imports`), and with the bench on `--extra-search-path` against its types.
  The harness's tests skip where the bench is not importable.
- **A run is one directory** (`scorecard.layout`). `run.json` declares it once: the family, its
  environments, how many of each environment's worlds in their seeded order, pilots or scored, the
  rungs of lift tests, this engine's arms and the tools' arms read beside them, a smoke flag, and
  for a scored run the bench's commit it is pre-registered against. Every later step reads the
  declaration, and declaring the run otherwise is refused. Exports, records and scores sit under
  it at paths named by family, environment, rung and seed; every file is written aside and moved
  into place.
- **An arm fits once and plans each variant** (`scorecard.arms.base.Arm`): a plug-in, a
  posterior's expectation and a tail's are one fit's cost and several records. `records` turns an
  arm's plans into the bench's record: the plan and the export's digest, the claim of the plan's
  gain over the status quo with its 90 % interval, the flags, the settings, what the variant does
  not give and why (`n/a`), the versions and the cost, the fit's and the plan's. A fit that raises
  fails every variant, a plan that raises its own, each with its traceback; the bench scores a
  failure at the status quo and counts it. A flag that is not a number is recorded as null, since
  a record is strict JSON.
- **One world, one process** (`scorecard.run`). `arm` runs each world an arm has not recorded in a
  process of its own, a few at a time, each stopped at a timeout with its whole session and, where
  asked, run in a systemd scope capped in memory with no swap. A world is done when its records
  exist, whatever the exit code, since a process killed for memory can exit 0; one that ends
  without them runs again, and after the last attempt it is recorded as a failure naming how each
  attempt ended. `score` scores every world whose records are all in, against its archived export,
  and appends its score; a world already scored is not scored again. `report` reads the largest
  prefix that every environment holds whole, so a run under way is read as a sequential look
  reads it.
- **Each world's process computes on one thread**, whatever the caller's environment: the harness
  sets `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `PJRT_NPROC` to 1 for it
  (`scorecard.provenance.ONE_THREAD`). The workers are the parallelism, so a world's cost is one
  core's and its arithmetic does not depend on the machine's count of cores. `OMP_NUM_THREADS`
  alone holds NumPy's BLAS but not XLA, whose CPU client sizes its pools by `PJRT_NPROC` and
  otherwise keeps a thread a CPU: on a CPU of 16 threads a jitted product held 16 compute threads
  with `OMP_NUM_THREADS=1` set, and one with `PJRT_NPROC=1`.
- **An arm that reads no lift test is handed none** (`Arm.tests`, `scorecard.arms.base.withheld`):
  each export without its tests and at rung 0, which is one observation at every rung of a world,
  checked by its digest. So the harness fits such a world once, in one process, and writes the
  first rung's records as each rung's, each naming its own export's digest and pointing its cost
  to the first.
- **A scored run refuses to start** unless the bench is at the declared commit, both trees are
  clean and JAX runs in double precision; a pilot or a smoke run records the same provenance and
  refuses only single precision. Each record carries the provenance.
- **What the arms read of an export** (`scorecard.arms.panel`). One national geo's weekly panel,
  dated from the bench's scripts' first Monday, the controls by their exported names. A lift row
  is read as the engine reads a test's analysis (`LiftEstimate`): the history's spend up to the
  readout's end with the test's dark weeks at 0, against the history's own; a readout of the dark
  weeks and the cooldown, both of which the export now names; the lift the row's weekly change
  times the dark weeks and its standard error likewise; the whole market, since a test is two
  universes at its scale. The row's spend is checked against the history's over its dark weeks.
- **A gain is computed in the harness from the engine's readings**: each reading's worth of the
  plan less its worth of the status quo, the history carried in, as `chc.allocation` reads a
  channel. The engine plans against its own reference (`PosteriorPlan.current_gain` reads the
  last periods' mean spend), and the bench's status quo is another.
- **The arms at the first milestone.** `bayes`: the static model's Hill and `tanh` fits with every
  lift row as an estimate, stacked; `stacked` plans for their expected return on the grid,
  `cvar 0.1` for the worst tenth of 400 readings against the status quo, the engine's no-test
  plan. `dlm`: the dynamic model over media searched for each channel's transform and the
  discounts (ADR 0020), media coefficients held at or above 0; `dlm plug-in` on the best spec's
  channels at the last week, `dlm mixture` and `dlm cvar 0.1` on 400 readings over the kept specs.
  The search starts at a retention of 0.5, the half-saturation point at the mean positive spend and
  a `tanh`, so the arm reads no other fit, and it reads no lift test: its records at every rung
  are one fit's. ADR 0038: it reads them, one fit a rung, and `dlm-untested` is the
  arm handed none.

## Consequences

- The harness's tests check the reading of an export against the bench's own construction: each
  lift estimate's spend reproduces the noise-free gap the world made over its readout for the
  world's own channel, to `1e-12` of it; its lift and standard error are the test's summed gap and
  noise scaled to the readout, to `1e-12`; a plan's worth is `chc.allocation.allocate`'s on a box
  of one point, to `1e-12`. Eleven mutants of the reading, the records, the scoring, the failure
  path and the timeout each fail a test, and so do eight of the one thread and of the tests
  withheld: a world's process counts its own compute threads.
- Neither arm forecasts the quarter's sales yet: the static model draws no posterior predictive
  over it and the dynamic model forecasts no fit whose effects carry transforms. Both records say
  so, and the forecast's axis reads them as absent, counted, not imputed.
- Each arm records its fit's response once, for the bench's return axis
  (`scorecard.arms.mapped`): `bayes` under `stacked`, `dlm` under `dlm mixture`, and the other
  variants name where. The dynamic arm's 400 draws are spread over the fitted specs as its
  readings are, from a stream of their own, each a draw of its spec's signed posterior with its
  coefficient's path over the history's weeks, and the bench holds them to the filter's own parts
  (`chc.dlm.decompose`) week by week. The static arm's readings are held in total to each draw's
  return times the history's spend, the identity the model is parametrised by; their weeks are
  their own. A dynamic record is about 3.7 MB, and it is written once for each rung.
- The dynamic arm draws each fitted spec's signed coefficients from four chains of 1 000 after
  1 000 of warm-up, where 400 readings over tens of fitted specs read about a dozen draws a spec;
  its fit spends most of its time there. The smoke run's worlds kept 23 to 55 specs each, the full
  grid's over a thousand, of which the arm fits 32 (ADR 0023); and fewer draws would not do: the
  signs bind, and 400 effective draws take about 4 000 (ADR 0022). A fitted spec whose chains have
  not mixed is drawn again (ADR 0022), and the arm's flags count those that still have not.
- A family with a geo axis is not read yet; its arms come with the geo engines.

## Alternatives considered

- **The harness in the bench.** Rejected: the bench is public and runs every arm as a black box
  over exports; the engine is private and its arms read its own fits.
- **One process for a whole run, a pool of workers.** Rejected: JAX holds compiled programs and
  their memory for the life of a process, a worker killed for memory takes its queue with it, and
  the exit code of a killed process is not evidence of anything.
- **The tests read as steady-state lifts**, as the reduced row states them. Rejected: the engine's
  likelihood reads a test through the carryover the change cost, which is what makes a test
  identify the carryover.

## Amended 2026-11-06: the pre-registration's hash, the bench's columns, and families 17 to 20

- **A run names the pre-registration it runs under** (`Declaration.preregistration`): the file's
  path and the SHA-256 of its bytes when the run is declared (`scorecard.layout.stamp`), kept in
  `run.json` in the form of the bench's stamp (`causaldyn_bench.scorecard.preregistration.Stamp`).
  A file in the engine's tree is named by its path in the tree, as
  `scorecard/preregistration/drift.md`, so the stamp names the file the same way on every
  machine; a file outside it is named by its absolute path. The runner declares a scored run only
  with one (`scorecard.run.declare`); a pilot or a smoke run may name one. The hash is the one the
  bench's signed tag makes public before the first scored world is drawn.
- **The report holds the run to its file** (`Declaration.preregistered`). It prints the stamp
  below its first line, as the bench prints it, and it refuses, before it prints a line, a run
  whose file is gone or whose hash moved after the declaration. A run whose file moved is also
  refused when it is declared again, as a run declared otherwise.
- **Its export and its arms do not start on a file that moved** (`scorecard.provenance.require`):
  any run that declares a pre-registration, pilot or scored, is refused there before it spends
  compute, every reason named. The export now passes the same check as the arms, so it also needs
  JAX in double precision, and for a scored run the bench at the declared commit and both trees
  clean: a world drawn otherwise is not the world the bench archives.
- **A joined root runs under the run's pre-registration** (`scorecard.claim.read_run`): the claim
  refuses a joined root that declares another one, or none beside one, naming both.
- **A run declared before this names none,** and it is read, written and reported as before, to
  the byte. The claim reader's pinned report is a scored run of that kind. The requirement sits in
  the runner's `declare` and not in `Declaration`, beside the bench's commit, so that such a run
  is still read; the bench's report of its family refuses it (below).
- **The bench's columns** (`report --columns`, `report --json PATH`). The report ends with the
  bench's report of the run's family on the run's scores (`causaldyn_bench.scorecard.report`):
  each arm's regret per euro, its CRPS and MAPE, the coverage of its ROI and mROI at 80, 90 and
  95 %, its effect shares against the spend shares, and the compute table. `--json` writes that
  report's JSON to `PATH` and its markdown beside it, the pair the bench keeps under `results/`,
  with the stamp in both, named by the declared path. Asked for neither, the report is as it was.
- **The bench's families 17 to 20 are in `FAMILIES`**, so that a run of any family the bench
  declares is declared, exported and read here:

  | Family | Bench module | Environments | What the arms read |
  |---|---|---|---|
  | 17 | `twins` | `curve`, `line` | two channels, `tv` and `search`, a kernel of one week, no control |
  | 18 | `long_term` | `unread`, `unbuilt` | the control `searches`, a brand measure |
  | 19 | `flat` | `flat-pla`, `flat-meta`, `flat-tv` | one channel whose spend never moves |
  | 20 | `promotions` | `independent`, `tv-0.5`, `tv-0.8` | the controls `promotion` and `price`, promotions following television |

  Family 2's new environment, `search-demand`, exports the control `queries`. The arms read a
  control by its name (`scorecard.arms.panel`), so neither control needs a change in an arm. In
  smoke mode, one pilot world of `search-demand` and one of `unread` ran through `bayes`, `dlm`,
  `ridge` and `chc`, each reading its control. One world each of `curve`, `flat-tv` and `tv-0.8`
  ran through `bayes`, `ridge` and `chc`. Every arm recorded its plan and was scored.

## Amended 2026-11-07: the calibration loop, the decision tools, the geo readers and the geo DLM

- **Seven arms join `scorecard.run.ARMS`**, each a module under `scorecard/arms/`, each run on one
  smoke world of its family end to end and scored. A run declares an arm by its name; its records
  are kept under its variants' names, which a pre-registration names:

  | Arm | Variants | Family | Reads the tests | ADR |
  |---|---|---|---|---|
  | `voi` | `voi`, `voi meridian`, `voi no test` | 0 | yes | 0077 |
  | `decisions` | `flighting`, `scenarios`, `plan table` | 0 | yes | 0078 |
  | `read-tbr` | `read tbr` | 8 | yes | 0078 |
  | `read-sc` | `read sc` | 8 | yes | 0078 |
  | `read-sdid` | `read sdid` | 8 | yes | 0078 |
  | `switchback` | `switchback` | 8 | yes | 0078 |
  | `geo-dlm` | `geo dlm` | 8 | no | 0078 |

- **The static arm's fit is shared** (`scorecard.arms.bayes.fitted`, `family_fits`): the
  calibration loop, the decision tools, the readers and the switchback each plan from the static
  arm's two fits of what they read, and its record. `family_fits` reads the estimates it is
  given in place of the export's rows; given none, it reads the rows, and `bayes`'s records are
  unchanged.
- **The geo readers and the switchback read universes the export does not hold**: an adapter
  draws them again from the world (`scorecard.arms.universes`), held by a test to the export's
  rows, until the bench exports them (ADR 0078).

## Amended 2026-11-07: version-4 exports of the geo families, and scores beside the bench's

- **A geo family is exported at the bench's version 4** (`causaldyn_bench.scorecard.observe.
  UNIVERSES`, `scorecard.run.export_all`): version 3's arrays, each test's universe at the rung,
  and the world's switchback test. Families 8, 9 and 10 are the bench's geo families
  (`causaldyn_bench.scorecard.geo.Geo`); the other families' exports are version 2, as before. A
  version-4 export has a digest of its own, and gives version 2's when a record names version 2.
  A run exported before keeps its archive: the bench keeps an export of another version where it
  holds the same world by that version's rule.
- **Every arm but the readers and the switchback reads version 2's arrays**
  (`scorecard.arms.panel.second`): an arm that reads no test is handed the export without its
  tests, the universes and the switchback test among them, so its observation is the same at every
  rung; a market is read from version 2's arrays, the geo export kept beside it. The records still
  name the export's version and digest.
- **A tool is handed a version-4 export as version 2** (`scorecard.run._hand`), written beside its
  record in place of the hard link. The tools' scripts read versions 1 and 2, and record version 2
  beside the digest of the file they read, so their records are scored on the archive by version
  2's digest. A hard link of the version-4 file would have its digest recorded under version 2,
  which the bench's scorer refuses.
- **Two variants are scored on the world's truth beside the bench's score** (`scorecard.run.
  SCORED`): `flighting`'s schedule and `switchback`'s reading (ADR 0078's amendment). The harness
  draws the world's truth a second time for a world that holds either; the score's line keeps the
  bench's form, the new numbers in the variant's record. The report prints them beside the
  variant's regret.
- **The adapter is removed**: the readers and the switchback read what the export holds (ADR 0078).
