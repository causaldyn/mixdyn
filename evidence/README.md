# Evidence

The scripts that ran the measurements some ADRs quote, and the records they wrote. Each ADR names
the file here that holds its number.

The scripts are kept as they last ran. Two kinds of change were made when they moved here, neither
touching a number: their imports were sorted to this tree's lint rules, and their usage lines (and
the provenance string `voi_bruteforce.py` writes) name this directory. They are not part of the
package: the wheel holds `src/mixdyn` alone, pytest does not collect them and ty does not check
them.

## Running them

From the repository's root, as each script's docstring shows:

```bash
uv run --no-sync python evidence/<script>.py ...
PYTHONPATH=<bench>/src uv run --no-sync python evidence/bench/<script>.py ...
```

- A script imports its neighbours in this directory, which Python finds because it runs the file
  from here.
- The scripts under `bench/` read the bench (`causaldyn_bench`, through `PYTHONPATH`) and its
  committed results, `results/track_m2_budgets*.json` and `results/track_m2_external*.json`, which
  stay in the bench's repository. Each record names the bench commit it ran at.
- The Stan oracles (`stan_bayes_mmm.py`, `stan_nested.py`, `stan_pathfinder.py`,
  `voi_bruteforce.py`) take a CmdStan install as their first argument. They ran on CmdStan 2.38.0;
  CmdStan is never a dependency.
- A record is reproduced at the mixdyn commit it names, not at today's: the engine has moved since.
  Each `report` subcommand reads a record and prints the numbers the ADRs quote. On 2026-11-06 every
  script here imported at mixdyn `9c36b9e`, and the `report`s of `dlm_media_coverage.py`,
  `dlm_media_coverage_per_effect.py`, `nested_validation.py`, `bench/track_m2_ridge.py` (both of its
  records below), `bench/track_m2_bayes.py` and `bench/track_m2_starts.py` printed the ADRs'
  numbers from the records here. `stan_pathfinder.json` and `plan_exactness.jsonl` were read
  directly: the medians and counts ADRs 0012 and 0034 quote are the records'.

## What each file shows

| file | what it shows | cited by | run with, as the record states it |
| --- | --- | --- | --- |
| `bench/track_m2_ridge.py` | the ridge engine's pilot of four calibrations on Track M v2, and its pre-registered scored run | ADR 0008 | — |
| `bench/results/track_m2_ridge_pilot.json` | the pilot: the joint score chosen over the tests in each fold's fit alone, -0.066 [-0.107, -0.024] | ADR 0008 | mixdyn `ff9c8b1`, the bench `e747207`, CHC `0ab2cd7` |
| `bench/results/track_m2_ridge_score_pilot.json` | the scored run's three arms on the pilot seeds, from which its pre-registration was written | the script's docstring | mixdyn `39a0a72`, the bench `e747207`, CHC `0ab2cd7` |
| `bench/results/track_m2_ridge_score.json` | the scored run: the library's search loses to CHC and PyMC-Marketing on both environments | ADR 0008 | mixdyn `39a0a72`, the bench `a901184`, CHC `0ab2cd7` |
| `bench/track_m2_bayes.py` | the Bayesian engine on Track M v2: the fit without tests, with them, and the posterior's expected return | ADR 0019 | — |
| `bench/results/track_m2_bayes_score_pilot.json` | its first pilot: the two planners ADR 0011 compares, the expected return less the plug-in +0.005 [-0.002, +0.012] on the drawn worlds and +0.028 [+0.004, +0.052] on the reference | ADR 0011's numbers; ADR 0019 | mixdyn `228f6ce`, the bench `a901184`, CHC `0ab2cd7` |
| `bench/track_m2_starts.py`, `bench/results/track_m2_starts_pilot.json` | the pilot that made Pathfinder's starts `fit_bayes`' default | ADR 0012 | the record names mixdyn `99f590d` with uncommitted changes; its fits ran on `f6eb787` (below); the bench `a901184`, CHC `0ab2cd7` |
| `stan_pathfinder.py`, `results/stan_pathfinder.json` | the engine's Pathfinder against CmdStan's on two normal targets | ADR 0012 | mixdyn `f6eb787`, CmdStan 2.38.0, seed 20261002 |
| `nested_validation.py`, `results/nested_validation.json` | the nested model's 90 % interval of television's return on 100 drawn worlds | ADR 0014 | mixdyn `99f590d`, seed 20261005 |
| `dlm_media_coverage.py`, `results/dlm_media_coverage.json` | the DLM over media's pre-registered coverage check, failed | ADR 0020 | mixdyn `5f9a413`, seed 20261010 |
| `dlm_media_coverage_per_effect.py`, `results/dlm_media_coverage_per_effect.json` | the same check with each effect's own discount, passed | ADR 0026 | mixdyn `0b349c6`, seed 2026100500 |
| `voi_bruteforce.py` | the refit oracle of a lift test's value, written to `tests/fixtures/oracles/voi_bruteforce.json` | ADR 0030, the fixture | CmdStan 2.38.0, seed 20261005, as the fixture records |
| `stan_nested.py` | the nested model as one Stan program, written to `tests/fixtures/oracles/nested_stan.json` | ADR 0033 | CmdStan 2.38.0, as the fixture records |
| `stan_bayes_mmm.py` | the one-geo Bayesian model as a Stan program, written to `tests/fixtures/oracles/bayes_mmm_stan.json`; `sbc_bayes_mmm.py`, `stan_nested.py` and `voi_bruteforce.py` import it | the fixtures | CmdStan 2.38.0, as the fixture records |
| `sbc_bayes_mmm.py` | the simulation-based calibration of the one-geo model; four scripts here import its `checkout`, which records the mixdyn commit a run read | — | its own records are not kept here |
| `bench/refresh_continuation.py`, `bench/results/refresh_continuation.json` | a refresh at the default weight against the full refit on Track M v2's worlds 900 continued by their quarter: each channel's mean log return 0.11 to 0.33 of the full refit's SD from it, against 0.01 to 0.10 seed to seed | ADR 0079 | mixdyn `33ed7a5`, the bench `ea24f2d`; CHC 0.22.0, which the record does not name |
| `bench/results/refresh_pilot.json` | the refresh cell's pilot (`python -m scorecard.refresh`): the stacked posterior's mean return never left the fit before's 80 % in 108 pairs of refits on six worlds; the ridge arm's left it on 5 or 6 of 6 | ADR 0079 | fitted at mixdyn `33ed7a5`, read at `0ab5849`, the bench `ea24f2d` |
| `bench/results/refresh_pilot_dlm.json` | the same cell's DLM arm on the four swing worlds, every fit in: its mixture's mean return left the fit before's 80 % in 4 of 72 pairs of refits, and moved by up to 19 a euro | ADR 0079 | mixdyn `d5a879f`, the bench `95a61f9`; CHC 1.2.0, which the record does not name |
| `plan_exactness.py`, `results/plan_exactness.jsonl` | the envelope's plan against the best split on the drift family's pilot | ADR 0034 | the record names no commit; it read a scorecard pilot's run directory, which is not kept |

Five records were written by an earlier revision of their script than the one kept here:

- `bench/results/track_m2_bayes_score_pilot.json` was written while `track_m2_bayes.py` planned
  the expected return with its own planner. The script now plans by the engine's `plan_posterior`,
  the change ADR 0019 made.
- The three `track_m2_ridge.py` records, the start pilot's and the Bayesian pilot's were written
  while the bench's harnesses read the mixdyn checkout when they wrote a record, after the run.
  They now read it when the run starts: a change to the provenance they write, not to what they
  compute. The start pilot's record shows why: it names `99f590d` with uncommitted changes, a
  commit made in the worktree while its fits ran on `f6eb787`.

Each record's provenance names the commit of each checkout it read, as the run wrote it. The local
paths the run wrote beside them were replaced by `<local checkout>` and `<venv>` before the
first public release (ADR 0095); nothing else in a record changed.

## What is not here

- The R scripts that recorded the R oracles (glmnet's and Robyn's). They stay in the author's
  private repository: this tree holds no R (`tests/test_python_only.py`). What they recorded is
  under `tests/fixtures/oracles/`, each fixture naming the package and version that recorded it
  (`tests/test_public_tree.py`).
- The harnesses no ADR cites, and their records.
