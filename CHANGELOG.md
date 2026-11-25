# Changelog

Each release's changes. A release's notes on GitHub are its section here, extracted by
`.github/workflows/release.yml`. The project follows the versioning rules of `docs/api/index.md`:
before 1.0 a minor release may change the public surface, and a patch release never removes or
changes a public name.

## [Unreleased]

### Added

- **The Bayesian models' options are public.** `mixdyn.models` lists the classes that
  `BayesSpec`, `BayesGeoSpec`, `BayesMedia`, `fit_bayes`, `fit_bayes_geo` and `fit_nested` take
  through their aliases: the return priors `RoiPrior` and `HalfNormalRoi`, the likelihoods
  `Normal`, `StudentT` and `LogNormal`, the curves `HillCurve` and `TanhCurve`, the starts
  `UniformStart` and `PathfinderStart`, and the coordinates `NaturalCoordinates` and
  `QRCoordinates`. `mixdyn.data` lists `ColumnFrame`, the frame `ColumnData` names. Each stays in the module that defines it, so no saved file's class moves.
  `tests/test_public_api.py` now follows a `type` alias to the classes of its value, which is how
  these had passed its check that a public name takes no class that no `__all__` lists.

### Changed

- **Every public name is documented.** The names `tests/test_public_names.py` listed as not
  documented yet carry a docstring or the string after their alias: `dumps` and `loads`,
  `CheckCode`, `Geo`, and the aliases `ColumnData`, `KpiUnit`, `Goal`, `Recommendation`, `Cost`,
  `Update`, `Verdict`, `When`, `Adstock` and `Shape`. The test's list of pending names is gone: a
  public name without documentation fails it.

## [0.3.0] — 2026-11-25

The rules the benchmark scored are public planners: `plan_margins`, `plan_anchored_margins` and
`plan_pooled_geos` plan what `margin held`, `anchored floored` and `geo eb` planned, to the last bit
(ADR 0102). The engines' fits and specs that public functions take are public, every module's
`__all__` is public, and a snapshot holds the surface's signatures. `stacked` and `lift_row_pick`
are renamed; the old names warn until 1.0.0. The benchmark page reads the claim arms' diagnostic
flags, post hoc, on M4's and M5's records. The reading changes no verdict. The margin planners set
a channel they fill to its ceiling exactly, which moves a plan by at most a unit in the last place. The
per-world records of M3, M4 and M5 are assets of this release.

### Added

- **The per-world records of M3, M4 and M5**, as assets of this release under CC BY 4.0: each
  run's scores archive, enough to re-run its reading, and its records in parts of a few families.
  Home-directory paths and the cloud box's hostname are replaced, so a file's SHA-256 differs from
  the one `evidence/results/` names; `MANIFEST.json` lists both, and `claim read --recorded` on the
  extracted roots prints each run's reading byte for byte.
- **The scored rules' public planners** (`mixdyn.decisions`, ADR 0102). `plan_margins` plans as
  `margin held` does, at k = 0; `plan_anchored_margins` as `anchored floored` does at its default
  `spend_floor=0.9`, and as `anchored held` with `spend_floor=None`; `plan_pooled_geos` as `geo eb`
  does. Each reads a `MediaPanel`, the channels' `Range`s and, where its rule reads them, the lift
  tests as `LiftEstimate`s, and returns a `MarginPlan` or a `PooledGeoPlan`, which `save` writes;
  `record` names the arm whose records the plan reproduces. Each refuses what its rule does not
  read, and its docstring states its arm's record. `tests/scorecard/test_public_planners.py` holds
  each to its arm on the arms' fixture worlds, plan, gain and interval to the last bit, and the
  reduction of a `LiftEstimate` to the bench's lift row bit for bit.
- **Public engine names.** `mixdyn.models` lists the fits and specs that public functions take,
  return or hold, with their `fit_*` functions: `BayesSpec`, `BayesFit`, `BayesMedia`,
  `BayesGeoSpec`, `BayesGeoFit`, `RidgeSpec`, `RidgeFit`, `Media`, `Control`, `LinearTrend`,
  `FourierSeason`, `Calibration`, `NestedSpec`, `NestedFit`, `Lifted`, `LiftReading`, `fit_bayes`,
  `fit_bayes_geo`, `fit_ridge` and `fit_nested`; the margin regression, `fit_margins`, `MarginFit`,
  `NoiseMixture` and `DegenerateFitError`; and the model over geos, `fit_pooled_geos`,
  `PooledGeoFit`, `Mixture` and `CellResponse`. `mixdyn.inference` lists `Diagnostics` and
  `HealthCard`. Each stays in the module that defines it, so no saved file's class moves.
- **`tests/test_public_api.py` and `tests/public_api.txt`** hold the public surface to a snapshot
  of every public object's signature, fields, members and value, the same on Python 3.12 to 3.15.
  Its writer refuses to drop a name before the release its package's `_DEPRECATED` names. The test
  also holds deprecated names to ADR 0062's removal rule, refuses a public name that shadows a
  submodule but the three that did, and a public name that takes, returns or holds a class no
  `__all__` lists.
- **`scripts/floor_record.py`** reads a scored run's scores into
  `evidence/results/floor_record.json`: at the tested rung, each family's paired difference in
  regret between `anchored floored` and `anchored held`, post hoc, with the SHA-256 of each
  `scores.jsonl` it read. ADR 0102 quotes it, and `tests/test_readme.py` holds the quote to it.
- **What the diagnostics tell (post hoc)**, a section of `docs/benchmark.md`. Each claim arm's
  flags, the plan's share of spend outside the logged range and each channel's, are read as
  classifiers of a regret above the status quo's, family by rung: their AUROC with its bootstrap
  interval and the rule's word, which keeps a flag only where its interval lies above 0.5. Beside
  them, every in-sample R² the records hold, Meridian's reviewer's and the ridge arm's, is read
  against the plan's regret. The flags read against a wrong return are not shown: they need `tau`,
  which no pre-registration has fixed yet.
- **`scripts/flags_record.py`** reads a scored run's scores into `evidence/results/m4_flags.json`
  and `m5_flags.json`, with the SHA-256 of each `scores.jsonl` it read and `"post_hoc": true`;
  `scripts/readme_record.py` writes the page's `flags` and `fit-regret` blocks from them.
  `tests/test_readme.py` fails where a block is not what the script writes, where a flag's word is
  not its interval's, or where a digest is not the one the run's record names.
- **`scorecard.flags.read`** (`scorecard/`, in the repository, not the wheel): each flag's row as
  `report --flags` prints it, its counts and its AUROC with its interval or the reason none is
  read. The report's lines and the record file are formatted from the same rows; the report prints
  as it did, to the byte.

### Changed

- **`uv.lock` locks causal-hybrid-control 1.5.0**, from 1.3.1. The dependency's floor stays
  `>=1.2.0`; the changes between them that the engine reads are opt-in or in planners it does not
  call, and the suite reads as before.
- **`GeoEbFit` and `fit_geo_eb` are `PooledGeoFit` and `fit_pooled_geos`**, renamed before either
  entered the format's registry, at the module path they had (`mixdyn.models.geo_eb`).
- **Every module's `__all__` is public** (ADR 0062, amended by ADR 0102), as the API index already
  read them.
- **The README's first plan calls `plan_margins`**, with public names alone, and its chart shows
  the plan's fill along the curves. The benchmark page's claim table names each arm's planner.

### Deprecated

- **`mixdyn.decisions.stacked` is `stacked_readings`, and `lift_row_pick` is
  `pick_by_lift_tests`.** The bare `stacked` is the scorecard's arm, and a lift row is the bench's
  export, not a public type. The old names import until 1.0.0, with a `DeprecationWarning` that
  names the replacement.

### Fixed

- **The margin planners plan a channel they fill at its ceiling, to the last bit.**
  `plan_corner` and `water_fill` (`mixdyn.decisions.margin_plan`) raised a channel to its ceiling
  as its spend plus the room left above it, `plan + (high - plan)`, which in floating point can
  miss `high` by a unit in the last place, either way: `0.7 + (2.9 - 0.7)` is
  `2.9000000000000004`. So a plan could lie past its box by that bit. A channel raised to its
  ceiling is now set to it. A plan moves by at most a unit in the last place, and only on a
  channel planned at its ceiling; every other channel's spend is as before, and a claim read off a
  plan moves with it in its last bits. Scored records are not read again: a run is read with the
  tree that ran it.

## [0.2.4] — 2026-11-24

Defects that a fuzz target of the readers of saved bytes found, and one beside them, each fixed
where the bad value is made. `serialize.load` refuses what it does not read with `FormatError`,
and `loads` with `SchemaError`, where each raised another error or read a value that its writer
then refused to write. A file that `save` wrote and a document that `dumps` wrote read as before.

### Fixed

- **`read_panel` names the values it cannot read as Python values.** An issue listed a value of a
  numpy column as numpy's `repr` writes it, `np.str_('soon')`, where the table holds `'soon'`.
  Dates and durations keep numpy's form, which `item` could turn into integers.

- **`load` refuses a file whose values a class's checks refuse.** A class is rebuilt by its own
  `__init__`, whose checks refuse a value with `ValueError`, and a value of another type than its
  field's with whatever error their code meets first. Only `TypeError` was read as the file's
  fault, so a `ValueError` or an `AttributeError` escaped `load` as it was.
- **`load` refuses an archive that Python's zip module does not read.** A member whose checksum,
  header or compressed data is damaged, or whose compression or encryption the module does not
  read, raised the module's or the decompressor's own error: `BadZipFile`, `zlib.error`,
  `LZMAError` and others. A zip version the module does not read raised `NotImplementedError`.
- **`load` refuses a manifest nested past Python's recursion limit.** `json` raised
  `RecursionError` on one nested past its stack, and the decoder on one it recursed through past
  the limit.
- **`load` refuses a node that holds what `save` never writes there.** A `$float` or a `$date`
  that is not text, a `$tuple`, `$list`, `$frozenset` or `$dict` that is not a list, a set's
  member or a mapping's key that no set holds, an enum's member that is not a name, and fields
  that are not an object each raised `TypeError`, `ValueError` or `AttributeError`.
- **`load` reads an array only as the `.npy` file that `np.save` writes, its header checked before
  it allocates.** `np.load` read an `.npz` archive as well, raised `EOFError` on an empty member,
  and allocated the shape a damaged header declared: 8 TiB from a member of a few bytes raised
  `MemoryError`. A JAX array of a dtype JAX does not hold raised JAX's `TypeError`.
- **`loads` refuses as not JSON what Python's parser does not read:** JSON nested past its stack,
  which raised `RecursionError`, and an integer of more digits than Python converts, which raised
  `ValueError`.
- **`loads` refuses a number that no float64 or int64 holds.** `Infinity`, which Python's parser
  reads although JSON has no such number, and a float past a float's range read as an infinity
  that `dumps` then refused to write. An integer past either range raised `OverflowError`.
- **`loads` refuses an array whose dtype is not text** with `SchemaError`, not `TypeError`.
- **`loads` refuses an array whose shape declares more values than an array holds.** The message
  that refused its data named the shape's product, and a product of more digits than Python
  converts raised `ValueError`.
- **`loads` and `load` name a value they refuse cut short.** A value nested deeper than `repr`
  writes, which `json` still reads, raised `RecursionError` as the message was written.
- **`loads` looks an enum's member up only by a value of a member's type.** The enum's own lookup
  named a value it does not hold with `repr`, so such a value nested deep enough raised
  `RecursionError` before the decoder could refuse it.

### Added

- **Python 3.15 among the package's classifiers.** CI has run the suite on it since 0.1.0; the
  package's metadata now names it beside 3.12 to 3.14.
- **A fuzz target of the readers of saved bytes**, `fuzz/fuzz_documents.py`. It reads documents
  and files, most of them ones the engine writes, damaged at random. `loads` must refuse each with
  `SchemaError` and `load` with `FormatError`, or return a value that `dumps` or `save` writes
  again and that reads back the same. `fuzz.yml` runs it with the other two.

## [0.2.3] — 2026-11-24

Three defects that the new fuzz targets found, each fixed where the bad value is made. The
scorecard's arms plan either on curves alone or on lines alone, so none of their plans moves.

### Fixed

- **`read_panel` refuses a day that no Python date holds instead of crashing.** numpy reads `-4`
  in a column of date strings as the year −4, and a `datetime64` can hold the year 10 000. Every
  issue lists its dates as Python dates, years 1 to 9999, so an issue that named such a row raised
  `AttributeError` or `TypeError` rather than `PanelError`. Such a value is now one of the
  column's values that are not dates.
- **`read_panel` reports a geo whose population is `None` as having no population.** It raised
  `TypeError`: the check for a missing population read only the geos that the mapping leaves out.
- **`water_fill` spends the budget where a line's slope does not read 1.** The price's
  bisection started at twice the highest margin times its slope at the floor, lines included,
  while the plan reads a line by its margin alone. Where a line's margin was above that start, the
  bisection never moved, and the plan spent more than the budget with no error. It now starts above
  every line's margin.

### Added

- **Fuzz targets** (`fuzz/`, in the repository, not the wheel or the sdist). atheris runs them,
  coverage-guided over libFuzzer, on branch coverage alone: in atheris 3.1.0 its comparison
  tracing keeps every numpy array that a traced comparison returns.
  - `fuzz_read.py` reads long tables of any content. `read_panel` must refuse each with
    `PanelError` or return a panel that keeps `MediaPanel`'s promises.
  - `fuzz_plan.py` plans boxes and budgets of any kind. `plan_corner` and `water_fill` must
    refuse each with `ValueError` or plan inside the box and on the budget, where no move of spend
    gains more than rounding.

  `fuzz.yml` runs each target for five minutes on every change and for half an hour a week, and
  `just fuzz` runs one locally. atheris is hash-pinned in `fuzz/requirements.txt` and kept out of
  the lock.

## [0.2.2] — 2026-11-24

M5's record joins M4's: M5's pre-registered claim held, 5 of 5 verdicts within δ = 0.05 on what
M4 left open, read on a third block of fresh seeds. With M4's 26 verdicts and M3's six geo
verdicts, the record holds 37 family-rungs; family 4 at k = 0 is the one national family-rung in no
claim, and its losses are stated. The engine's code is 0.2.1's: `src/` is
unchanged, so no fit or plan moves.

### Added

- **A third confirmatory run** (`scorecard/`, in the repository, not the wheel). A declaration's
  `third_confirmatory` draws each environment's worlds from a block of 1 000 seeds of its own, from
  seed 1 000 000 on, which no earlier run read (ADR 0101).
- **Charts of the record** (`docs/assets/`, drawn by `scripts/record_charts.py` from the runs'
  records and the README's example, light and dark): each family-rung against its closest rival
  with its interval, every comparison by rival, each arm's CPU time and peak memory over M4 and
  M5, and the first plan's margins and plan.
  `tests/test_readme.py` fails where a committed chart is not what the script draws.
- **M5's pre-registration and record.** `scorecard/preregistration/m5.md` (SHA-256
  `86e12d7cc1e2b8eae21689db069c5600f0fc8a9d64d797c2a9350ac2ba7970d1`), `m5.json` and
  `m5_sizing.py` are byte for byte as frozen on 2026-11-23, before any confirmatory world was
  fitted; the bench's signed tag `preregistration-m5` carries their hashes. M5 claimed the k = 0
  rungs of families 1, 6 and 14 and both rungs of family 15 (siMMMulator); all 5 verdicts held
  (4.997 expected). `evidence/results/m5_record.json` is its record as the benchmark page prints
  it, written by `scripts/readme_record.py extract --run m5` with the SHA-256 of every file it
  read.

### Changed

- **The README is short**: what the engine is, the chart of its record, a scoreboard of Hall's
  reader's words by rival over the 38 family-rungs, installation and a first plan. The full record
  moved to the documentation's benchmark page, `docs/benchmark.md`; where the engine sits moved to
  the documentation's home page.
- **The benchmark page states the record of M4, M5 and M3 together**: 37 family-rungs, each row
  naming its run, with M5's Holm's losses within δ on family 15, its versions and its compute. The
  combined tables' blocks are named `claim`, `cells`, `words`, `losses` and `exploratory`, and
  M5's own `m5-versions` and `m5-compute`; `tests/test_readme.py` checks M5's blocks and the
  prose beside them against its reading as it checks M4's.

## [0.2.1] — 2026-11-23

The README states M4's record: its pre-registered claim held, 26 of 26 verdicts within δ = 0.05 on
fresh seeds, with M3's six geo verdicts beside them and every loss stated. The engine's code is
0.2.0's: `src/` is unchanged, so no fit or plan moves. The scorecard's reader, in the repository and
not the wheel, reads the joined root's variants beside the claim and declares rungs past the sizing
cap as exploratory.

### Added

- **M4's pre-registration and the harness it ran on** (`scorecard/`, in the repository, not the
  wheel). `scorecard/preregistration/m4.md` (SHA-256
  `ec0df163332e40244ba0747ae1c6c9654664fc7354137a9498ad5d979e9e45b1`), `m4.json` and
  `m4_sizing.py` are byte for byte as frozen on 2026-11-21, before any confirmatory world was
  fitted; the bench's signed tag `preregistration-m4` carries their hashes. The harness adds a
  second confirmatory run, whose worlds are each confirmatory block's seeds past the first run's 300
  (`Declaration.second_confirmatory`, ADR 0098), and the `anchored` arm's variant
  `anchored floored`, the tested rung's spend floor, never routed (ADR 0099). Both ADRs were
  numbered 0096 and 0097 where M4 was frozen, and the pre-registration cites them by those numbers.
  No other arm's record moves.
- **A run declares its rungs past the sizing cap as exploratory** (`Declaration.exploratory_rungs`,
  ADR 0100). Such a rung is read and printed exploratory, and left out of the claim, Holm's list
  and the order of verdicts. Until now a declaration could keep a rung out only by naming every
  rival there predicted to fail, as M4's roots did for the k = 0 rungs of families 1, 4, 6 and 14:
  the reading printed "predicted to fail" where no failure was predicted and counted those 20
  comparisons in Holm's list. A family and rung read only as exploratory now stops the headline,
  as the pre-registrations state; a tool read on its cap and exploratory still does not. M4's
  records declare `predicted`, so their reading is unchanged.

- **The README states M4's record.** M4's pre-registered claim held: 26 of 26 verdicts within
  δ = 0.05 on 15 national families, read on fresh seeds (25.23 expected), with M3's six geo
  verdicts beside them, 32 family-rungs. Its losses, deviations and code differences are stated,
  and every table is generated from `evidence/results/m4_record.json` and `m3_record.json`
  (`scripts/readme_record.py`, checked by `tests/test_readme.py`).

### Fixed

- **`claim read --recorded` reads the joined root's other variants.** It joined only the arm that
  plans the default above 0, so M4's reading left out `anchored`, `anchored curved` and
  `anchored held` at the tested rung. Every arm of the joined root is now read beside the claim at
  the rungs above 0, on the worlds that score it. Read again over M4's 32 roots, no line of the
  reading moves and 225 lines are added.

### Changed

- **The digest tests no longer need a key for every new machine.** A machine whose dispatch key is
  not recorded passes a case where it draws a digest a recorded key holds for that case; a
  recorded key still holds its own digest only. XLA names every vector feature a host shows, so
  one CPU on two hosts gave two keys.

## [0.2.0] — 2026-11-21

The margin regression reads under the noise mixture by default. Without it, its ridge prior pulled
the sales' level into the channels' margins, which grew with the level: up to 5.5 times the truth
at a level of 20 000 on 20 seeds. Every fit that relied on the default moves; `noise=None` gives
the old fit to the bit, and the scorecard's arms pass it. The mixture's memory is bounded, and a
fit that would not fit in memory stops before it starts.

### Added

- **The README's record of M3, generated from the run's records**: a conflict-of-interest
  section; the error bars' definition and their source of variability; the complete cases behind
  every cell of the claim's table; the tuned rivals' reading, `meridian tuned` and `robyn tuned`
  beside Meridian and Robyn at their recommended settings, with their numbers; a versions table of
  every arm (each record's packages, Python, commits and settings); a compute table per family and
  per arm (CPU model, threads, CPU quota, peak memory, wall a process, core-hours); and a paragraph
  on where an MMM checked by experiments sits against multi-touch attribution, with its sources.
  `scripts/readme_record.py` reads the tables from the records into
  `evidence/results/m3_record.json` and writes them into the README, and `tests/test_readme.py`
  fails where a table differs from what it writes, or where a cell of the typed claim table drifts
  from the run's reading.
- **Build provenance on each release**: the release workflow attests the wheel's and the sdist's
  provenance with GitHub's attestation action, checks the bundle against both files before the
  upload, and attaches it to the GitHub release as `mixdyn-<version>.provenance.sigstore.json`.

### Changed

- **`fit_margins` and `fit_curved_margins` read under the noise mixture by default**,
  `noise=NoiseMixture()` (ADR 0097). Without it the ridge prior on the trend pulls the sales' level
  towards 0 and the channels' columns take part of it up: on the README's history, its sales 200
  noise deviations up, margins of 1.5, 3.0 and 0.5 read 4.48, 8.30 and 0.20; under the mixture
  they read 1.44, 2.19 and 0.21 at any level. `noise=None` fits as before, to the bit, and the
  scorecard's arms name what they were measured with, so no arm's plan moves. Every fit that named
  no `noise` moves. Under the mixture a history whose nuisance columns the weeks do not tell apart,
  a price that never moves among them, raises `DegenerateFitError`.
- **The noise mixture holds its fits' second moments only up to 1 GiB** (ADR 0097). Past that, or
  where the memory left is short, it reads every fit again for the mixture's second moment, a block
  at a time, to the same bits. A fit of 104 weeks peaks at 0.8 GiB over the interpreter at six
  channels and 1.3 GiB at seven, where it took 1.9 GiB and more than 8 GiB; eight channels take
  8.6 GiB by the fit's estimate. Where the memory left, the system's or a cgroup v2 limit's, cannot
  take the fit, it raises `MemoryError` before it fits, naming the channels, the memory and the
  remedies: fewer channels, a `NoiseMixture` of fewer error models, or `noise=None`.

### Fixed

- **The README said Meridian ran on a capped number of family 10's worlds.** Its cap there equals
  the family's worlds: it ran on every world, and is read beside the verdict because the cap is
  declared exploratory, as in M2.
- **A tanh curve's fit no longer depends on the platform's libm.** The curve's constant
  `atanh(1/2)` was `math.atanh(0.5)`, which glibc 2.39 (Ubuntu 24.04) returns one ulp below the
  correctly rounded `ln(3)/2` that glibc 2.43 returns, and a seeded fit's draws then differ by more
  than the last bit. It is now the correctly rounded literal in `models.bayes_mmm`,
  `transforms` and `inference.sensitivity` (ADR 0096). Under glibc 2.43 every number is unchanged;
  under glibc 2.39 the constant and every number built on it are glibc 2.43's.

## [0.1.1] — 2026-11-21

The project's public pages, and no change to the engine: `src/` is byte-identical to 0.1.0's, so
every fit, plan and number 0.1.0 gives, 0.1.1 gives.

### Added

- **A documentation site**, https://causaldyn.github.io/mixdyn/, built with mkdocs-material and
  mkdocstrings and deployed by each release from its tag: the README's example and measured record,
  the walkthrough of `SKILL.md`, the scope, and an API reference over every public module. A strict
  build of it runs in CI on every push. The decision records stay on GitHub.
- **`CITATION.cff`**, which Zenodo archives each release with; a test keeps its version and date
  equal to the package's and to this file's. **`SECURITY.md`**: reports go through GitHub's private
  vulnerability reporting, and every file on PyPI carries a PEP 740 attestation of the workflow that
  uploaded it. **`CODE_OF_CONDUCT.md`**: the Contributor Covenant 2.1.
- **README badges**: CI, the PyPI version, the supported Pythons, the licence, the documentation
  site and the OpenSSF Scorecard, whose workflow now runs on this repository.

## [0.1.0] — 2026-11-20

The first public release. The engine's history before it is not public; this section lists what
the release holds.

### Measured

- **A pre-registered non-inferiority claim on the causaldyn bench (run M3).** On simulated worlds
  whose truth is known, mixdyn's plans are within δ = 0.05 of regret per euro of budget of those
  of Meridian 2.1.0, PyMC-Marketing 1.3.0, Orbit 1.1.5.1, the status quo and the equal split, on
  17 pre-registered family-rungs. Each verdict was tested at one-sided 0.025 in a fixed sequence,
  on fresh confirmatory seeds: 17 of 17 held, against 16.94 expected.
  - At k = 0 (history only) `margin held` has a lower regret than every rival on every
    confirmatory national family. At the tested rung (four go-dark lift tests) `gs held`, and on
    the geo families `geo eb`, are level with PyMC-Marketing and have a lower regret than the
    other rivals.
  - **Losses, stated with the claim:** S-curves with no lift test (family 4, k = 0) lose to
    PyMC-Marketing by +0.184 [+0.136, +0.232] and to Orbit by +0.171 [+0.118, +0.223] a euro, not
    within δ. At the tested rung, drifting effects (family 1) and truth outside the history's range
    (family 6) lose to PyMC-Marketing by +0.029 [+0.002, +0.055] and +0.039 [+0.022, +0.055], not
    within δ. Holm's step-down finds two losses within δ, families 3 and 5 at the tested rung
    against PyMC-Marketing (+0.008 and +0.020).
  - **No headline:** families 12 and 13 at k = 0, and family 12 at the tested rung, needed more
    worlds than the run's cap and are exploratory, so "within δ of every maintained MMM" is not
    said. Bench families 11, 16 and 21 were not in the run. PyMC-Marketing ran its static model.
  - The pre-registration is `scorecard/preregistration/m3.md` (SHA-256
    `d684b25ccae0e3574fe2f962097e399c9f400f0641e52431cf794e131763d924`), frozen on 2026-11-17 in
    the then-private repository: there is no public time-stamp before the run. The README states
    the full table, every loss and what the run did not cover.

### Added

- **Data** (`mixdyn.data`, `read_panel`): a weekly or daily panel of a KPI, media spend, controls
  and levers, national or over a geo hierarchy, read from dicts, pandas or polars frames and
  checked, every finding an issue with a code; national media allotted to geos by population;
  recurring events read off the panel's own records; the flag for spend that reads the KPI
  (ADR 0074).
- **Transforms** (`mixdyn.transforms`): adstock kernels and saturation curves in two families,
  each in its engine's convention (ADR 0064).
- **Engines** (`mixdyn.models`):
  - the ridge engine, its search over transforms and its calibration by geo tests (ADR 0008);
  - the discount dynamic linear model over media, with its decomposition, diagnostics, discount
    choice and lift tests read by the filter (ADRs 0020, 0038, 0066);
  - the Bayesian engine, one geo and over geos, with returns as parameters and lift and geo tests
    in the likelihood, sampled by the engine's own NUTS started by Pathfinder (ADRs 0009 to 0015,
    0018, 0036);
  - the nested model, media through a mediator to the KPI (ADRs 0014, 0033);
  - the margin model: each channel's margin by regression under a mixture of error models,
    linear or curved, with empirical-Bayes shrinkage (ADRs 0086, 0087, 0091);
  - the closed-form model over geos fitted by empirical Bayes (ADR 0090);
  - a demand proxy, query volume for paid search, and the one role a model gives it (ADR 0074).
- **Inference** (`mixdyn.inference`): NUTS, Pathfinder, convergence diagnostics with thresholds
  read where Stan reads them (ADR 0065), PSIS-LOO and the stacking of models' predictive
  distributions, the health card (ADR 0060), and sensitivity to an
  unobserved confounder: Cinelli and Hazlett's robustness values and the marginal sensitivity
  model's `Gamma*` (ADR 0061).
- **Experiments** (`mixdyn.experiments`): geo-test readers (difference in differences, time-based
  regression, synthetic control, synthetic difference in differences), a switchback reader, geo
  designs randomised within matched pairs with their power, and the value of a lift test to the
  plan and the set of tests chosen by it (ADRs 0030 to 0032, 0047 to 0049, 0055, 0056).
- **Calibration** (`mixdyn.calibration`): lift estimates and their tables, and margins anchored on
  lift tests (ADRs 0015, 0088, 0089).
- **Decisions** (`mixdyn.decisions`): one definition of each return, plans for a posterior's
  expected return, its worst share and a target return, the no-test plan against the plan in
  place, flighting, scenarios, the plan table, geo plans, and the margin rule's plans with its
  observed-range box and time guard (ADRs 0016, 0019, 0029, 0034, 0044, 0050 to 0052, 0071, 0092).
- **API** (`mixdyn.api`): a factsheet of a fitted model, the decision on one page, the estimands of
  the tests read, a file format that pickles nothing (`save`, `load`, `FORMAT_VERSION` 1) and
  versioned JSON documents (ADRs 0062, 0063, 0072).
- **The scorecard's harness** (`scorecard/`, in the repository, not the wheel): declares a run,
  exports the bench's worlds, runs every arm, this engine's and the rivals', scores every record
  and reads the pre-registered claim (ADRs 0021, 0039, 0040, 0080, 0081). It needs the bench
  (`causaldyn-bench`) on `PYTHONPATH`.
- `SKILL.md`, an Agent Skill that walks one panel through read, fit, diagnostics, tests, plan and
  report, each step gated by the engine's own checks; `scripts/real_data_protocol.py`, the
  protocol a partner would run to score predictions of withheld lift tests on real data.

### Packaging

- On PyPI as `mixdyn`, Apache-2.0 (`License-Expression: Apache-2.0`, with `LICENSE` and `NOTICE`
  in the wheel and the sdist), contributions under the Developer Certificate of Origin
  (`CONTRIBUTING.md`; ADRs 0054 and 0095).
- Python 3.12 or later; CI runs the suite on 3.12, 3.13, 3.14 and 3.15.
- Runtime dependencies: `causal-hybrid-control>=1.2.0`, `jax>=0.4.30`, `numpy>=2.3`,
  `scipy>=1.16`; `uv.lock` locks causal-hybrid-control 1.3.1. The packages the engine may install
  at run time are an allow-list that CI checks against the lock (ADR 0001).
- The wheel holds `mixdyn` alone; the sdist leaves out `evidence/` and `scorecard/`.
- Releases are published from a signed tag through PyPI's Trusted Publishing, with no stored token.
