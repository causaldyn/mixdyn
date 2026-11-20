# 0039. One runner for every arm

Status: accepted, 2026-10-11; amends ADR 0021's "one world, one process"; amended 2026-11-16 by
ADR 0084: three tuned tools' arms, and the bench's options their scripts take

## Context

ADR 0021's harness runs this engine's arms one world a process. Each process runs in a scope
capped in memory, is stopped at a timeout and gets two attempts. A world that still has no
records is then recorded as a failure: it plays the status quo and names how its process ended.

The tools' arms, the bench's own scripts, ran outside the harness, from a shell queue. The queue
split each tool's worlds into shards. A shard was one process that fitted its worlds in turn
under the tool's memory cap. In the drift run this lost worlds in three ways:

- One world that outgrew its cap killed its shard's process. The shard's later worlds were not
  tried until the queue's second pass, and the killed world left no record at all.
- A world with no record never reaches `scores.jsonl`, and the report reads only the look that
  every environment holds whole. So one tool's killed world stopped every arm's reading at that
  world's place in its environment's order. Failure records for such worlds were written by hand
  after the run.
- The Robyn queue had no second pass, so one segfault in R lost a world until it was run again by
  hand.

Three more defects were found while the tools were brought into the harness:

- A tool's script writes its record in place, not aside. A process killed while it writes leaves
  a record cut short, and `score` failed on it.
- `uv` runs every Python tool. When its child is killed by signal `n`, `uv` ends with exit code
  `128 + n`, as a shell does: on uv 0.13.0, 137 for a child killed by its scope's memory cap and
  139 for one killed by SIGSEGV. A process the harness starts directly ends with `-n`. `ending`
  read only a negative code as a signal, so it would have recorded a tool's cap kill as `exit`.
- The harness killed a process's session only at its timeout. A process that ends otherwise can
  leave what it forked running. R's forked workers outlive R's own segfault: 20 hours after the
  drift run's Robyn segfault, four of its workers were still alive, each holding about 0.57 GiB
  resident, outside every gate's account.

## Decision

- **Every arm runs through `scorecard.run arm`, by one rule**, this engine's and the tools'
  (`scorecard.tools.TOOLS`). Meridian, PyMC-Marketing in its static, shared and channel settings,
  and Orbit run as `uv run --no-project --python 3.12 --with-requirements scripts/<arm>.txt
  python scripts/<arm>.py` from the bench's tree. Robyn runs as `Rscript --vanilla
  scripts/robyn_arm.R --ranges genre`, on the R library and nevergrad's environment that the run
  declares. The bench's scripts do not change.
- **One world, one process.** A tool's script gets a directory of its own for one world at one
  rung (`inputs/`), as both its `--worlds` and its `--out`. The directory holds the world's
  export, hard-linked, and nothing else. A tool reads the lift tests, so each rung is a process
  of its own.
- **A tool's record enters the run only once it parses.** The harness moves it from the world's
  directory into `records/` with one rename. A record that does not parse is set aside beside the
  world's log, and the world runs again. `score` parses every record of a world before it scores
  the world. A world with a record that does not parse is not scored: it waits for its arm.
- **A relaunch runs only what is missing.** It takes a whole record that a process made before
  the runner stopped. It sets aside a record in the run that does not parse. It runs only the
  worlds that then have no record.
- **Each process runs in a systemd scope of its own**, capped at its arm's declared memory with no
  swap. It is stopped after its arm's timeout. At its end or its timeout, its whole session is
  killed, so what it forked ends with it. The harness waits on the process through a file
  descriptor (`pidfd_open`) that does not reap it, so the session is still the process's own when
  it is killed.
- **The run's memory gate admits each process.** The gate is a lock on `logs/gate.lock`, so it
  admits one start at a time, whichever arm and whichever runner starts it. A process starts only
  when the memory available (`MemAvailable`) holds its cap and the declared reserve. Its start
  then holds the gate for the declared settle time, or until the process ends, so that its memory
  shows before the next start.
- **Two attempts, then the failure record.** A world that ends without its records runs again, up
  to the declared attempts. Then it is recorded as a failure of the kind that its last ending
  gives: the same record for a tool as for this engine. `ending` reads an exit code of `128 + n`,
  where `n` is a signal's number, as that signal.
- **The run declares how it runs**, in `run.json`, so that a run says what it ran: each arm's
  memory cap, timeout and slots (`Limits`); the gate's reserve and settle time; the attempts;
  Robyn's R library and nevergrad's environment; and the directory of family 15's worlds.
  `declare` refuses an arm without limits, and Robyn without its two paths.
- **Threads.** Every process of a run, the tools' included, runs with `ONE_THREAD`, which now also
  sets MKL's thread count to 1, as the drift run's queue set it for the tools. Robyn keeps its 4
  cores: its script forks 4 workers, which these variables do not limit.
- **Families 14 and 15**, Robyn's home ground and siMMMulator's own worlds, are in `FAMILIES`.
  The bench reads family 15's worlds from files and does not draw them, so its entry is its class,
  made on the directory that the run declares.

## Consequences

- A tool's killed world leaves a failure record in the run, so one world no longer stops the
  report's look. The tests run a fake tool whose worlds record, record an error, exit with code 3,
  are killed by a 256M cap, segfault, die halfway through writing their record, and sleep past
  the timeout. Each world ends as a record, or as a failure of its kind, and the report reads
  every world.
- A tool's world pays its interpreter's start, and for a Python tool `uv`'s check of its cached
  environment, once a world rather than once a shard.
- A run's limits cannot change after it is declared: a world ran under what the declaration
  says. A run that needs other limits is a new declaration.
- The gate admits one run's processes. Two runs at once each have their own gate, and both read
  the same memory available.
- A declaration written before this decision still reads, with no limits. `arm` refuses to run
  its arms.

## Alternatives considered

- **Keep the shell queue and add a failure writer**, which writes the engine's failure record for
  each world that a tool left without a record after the queue's passes. Rejected: a run then has
  two rules, a shard still loses its later worlds when one world kills it, each caller of the
  queue must remember the writer, and a record cut short is still read as a record.
- **Shards with resume inside the shard**: each shard notes the world it starts and, when it
  starts again, skips a world that killed it. Rejected: each of the bench's scripts must change,
  in Python and in R; a killed world still stops every later world of its shard until the next
  start; and the cap, the timeout and the failure kind still belong to a process of many worlds,
  not to one world.
