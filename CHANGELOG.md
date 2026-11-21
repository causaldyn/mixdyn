# Changelog

Each release's changes. A release's notes on GitHub are its section here, extracted by
`.github/workflows/release.yml`. The project follows the versioning rules of `docs/api/index.md`:
before 1.0 a minor release may change the public surface, and a patch release never removes or
changes a public name.

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
