# 0035. A record's compute, and a plan whose chains are past their bounds read twice

Status: accepted, 2026-10-06; amended 2026-11-06

## Context

The scorecard's cost axis asks for the time to fit and to plan, peak memory and the failure rate.
A record of the engine's (ADR 0021) held each step's wall and CPU seconds, and a failure's message,
which plays the status quo. It did not say:

- how much of a fit is JAX compiling, which a fresh process a world pays in full and a process
  that lives on pays once;
- how much memory a world takes, which sets the memory cap and the workers;
- how fast a sampler gains effective draws: their counts were flags, apart from the seconds;
- what kind a failure is, except in its message.

Every arm that samples was scored on its plan whatever its chains said, and no rule said whether
that is the reading. In the drift family's pilot, 120 world-rungs at k = 0 and 4, these plans
read a fit whose largest R-hat is above 1.01 or whose divergent transitions are above 27 in 4 000
draws:

| arm | plans |
|---|---|
| `stacked` and `cvar 0.1` (one fit) | 15 |
| `dlm plug-in`, `dlm mixture`, `dlm cvar 0.1` | 0 |
| PyMC-Marketing, static | 62 |
| PyMC-Marketing, one HSGP shared | 78 |
| PyMC-Marketing, an HSGP a channel | 89 |
| Meridian | 59 |

## Decision

- **A record's cost holds its compute** (`scorecard.arms.base.records`):
  - each step's seconds compiling, `fit_compile_seconds` and `plan_compile_seconds`: the
    durations of the events JAX records for tracing a function, lowering it and XLA's compiling
    it (`COMPILES`), summed over the step;
  - the process's peak resident set, `VmHWM`, in bytes: `fit_peak_rss_bytes` once the fit
    returns, and `peak_rss_bytes` once the last variant is planned, the world's;
  - where a plan reads a Markov chain's draws, `ess_bulk_per_fit_cpu_second`: the least bulk
    effective sample size of those chains over the fit's CPU seconds. The static arm's chains are
    the fits whose draws its readings hold (`bayes.shares`). The DLM's plug-in reads the best
    spec's chains, whose means its channels are, and its other plans every fitted spec's.
- **A failure names its kind** (`failure_kind`): `fit`, `plan`, or `record` where a plan is not a
  finite spend for each cell; and for a world's process that left no records, by its last
  attempt, `timeout`, `signal` (killed, as by the memory cap) or `exit`. The report counts each
  arm's failures by kind; a tool's script names none, and its failures count as `raised`.
- **A plan whose fit's chains are past their bounds is read twice** (`scorecard.report.past`,
  `gated`): first on its plan, as every plan is; then at the status quo's regret, as a failure
  is. The table, the arms' regrets and the engine's paired comparisons are given both ways
  wherever a plan of the look is past them.
  - The bounds: a largest R-hat above 1.01, ADR 0024's ceiling on its normal and `tanh` worlds,
    every sampled arm's noise here being normal; or divergent transitions above 27 in 4 000 of a
    fit's draws, ADR 0025's ceiling at its most, Stan's run on its `tanh` world.
  - Read on the static arm's families whose draws the readings hold, and on PyMC-Marketing's and
    Meridian's one fit. The DLM's chains are Gibbs chains, which do not diverge: its plan is past
    where a fitted spec's chains did not mix by CHC's own check, R-hat at most 1.01 and 100
    effective draws a chain, after ADR 0022 drew them again. Orbit's fit is variational and is
    not read.
  - A sampler's flag that is not a number is past. A record that names an R-hat in another form
    is refused: the bounds are not stated for it.
- **A claim is read on the first reading**, and the second is reported beside it. A claim is
  about the plan a procedure gives its user, and each tool gives its plan whatever its chains say,
  with a warning at most, as the engine does. At the status quo's regret, an arm whose chains
  fail would be scored on the status quo's plan, not its own.

## Consequences

- *On the pilot*, the second reading moves the mean regret per euro over the 120 world-rungs
  from 0.254 to 0.272 for PyMC-Marketing's static model, 0.226 to 0.316 with one HSGP shared,
  0.297 to 0.370 with an HSGP a channel and 0.276 to 0.318 for Meridian; and from 0.329 to 0.320
  for `stacked` and 0.367 to 0.365 for `cvar 0.1`. The status quo there reads 0.425, the worst
  of the arms, so the second reading raises the tools' regret and lowers the engine's: the first
  is not the engine's choice for its own sake.
- The pilot's records predate the compute: a run's records hold it from this commit on. The
  report prints it for the records that hold it, as recorded under the run's load: a number for
  sizing a run, not a timing, which is quoted only from a run on a quiet machine.
- JAX's events of one call do not overlap: a jitted function that calls another jitted function
  records one trace, one lowering and one compile.
- `VmHWM` and `getrusage`'s `ru_maxrss` read the same peak, but `getrusage` reads per-CPU
  counters it does not fold, and differed from `VmHWM` by about 1 MB of 350 MB in the harness's
  test; the test holds them within 5 %.
- *Tests*: the compute's fields and their sums, the compiling against JAX's own events, the
  fit's peak read before the plans, and the rate; a `NaN` ESS recorded as absent, not as a rate;
  each kind of failure; how a process's ending maps to its kind; the bounds on every record form
  at both sides of each bound, a family with no draws read, a null flag, Meridian's 10 000 draws;
  and the report's second reading, present where a plan is past and absent where none is.
- *Mutations*, each run on the tree in turn: 30, of the failure kinds (3), the rate (4), the
  compiling (4), the peak (2), the process's ending (2), the bounds and the second reading (11),
  the static arm's least ESS and shares (2) and the DLM's (2). Three survived a first pass: `VmHWM`
  read as 1 000 bytes a kB, inside the 5 % the test gave `getrusage`; the most of two families'
  ESS for the least, where the test's fit read one family; and a check that `past` returns None
  for a failure, which a failure's record, holding no flags, never needed. The test now reads the
  status's own line, the least ESS is tested on two families, and the check is gone. The other 29
  each fail a test.
- *Left*: the tools' scripts record neither compiling nor peak memory. Their processes run many
  worlds each, and PyMC's chains run in child processes, so a peak a world needs a process of its
  own there; their cost stays wall and CPU seconds.

## Alternatives considered

- **The second reading as the claim's.** Rejected for the reason above; it is reported beside.
- **The tools' own convergence checks**, Meridian's review among them. Rejected as the rule: one
  bound for every arm that samples, so the arms are read alike. The review stays in Meridian's
  flags.
- **ADR 0025's ceiling by curve**, 1 divergence in 4 000 on its normal world and 27 on its `tanh`
  world. Rejected: Stan's counts were taken on those worlds alone, and a curve does not carry them
  to another world; the largest is the bound that wrongs no sampler Stan itself would pass there.
- **`ru_maxrss` for the peak.** Not taken: `VmHWM` is the kernel's exact count, in the same unit.
- **Compile seconds from a second fit in the same process.** Rejected: a second fit doubles the
  cost of a world to measure what JAX already records.

## Amended 2026-11-06: where a record's process ran

- **A record's cost names where its process ran** (`scorecard.arms.base.records`): `cpu_model`,
  the CPU's model as the kernel names it, and `threads`, the process's `OMP_NUM_THREADS`, 1 under
  the harness (`scorecard.provenance.ONE_THREAD`). These are the two fields the bench's tool arms
  now record beside their seconds and their peak resident set, so the bench's compute table
  (`causaldyn_bench.scorecard.report.compute`) reads the engine's arms as it reads the tools'. A
  record whose rung points to another rung's fit still holds no compute, and counts in no column.
- Beside them, `pjrt_nproc`, the process's `PJRT_NPROC`, null where it is unset: the setting that
  sizes XLA's CPU pools, which `OMP_NUM_THREADS` does not (ADR 0021). `threads` keeps the bench's
  definition, so the table reads every arm by one; the bench's table does not read this field.
- The table states what was spent under the load the machine carried, not a speed.
