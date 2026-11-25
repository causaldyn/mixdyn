# 0102. The scored rules as public planners, the spend floor as a parameter, and the surface held by a snapshot

Status: accepted, 2026-11-25; adds `mixdyn.decisions.plan_margins`, `plan_anchored_margins` and
`plan_pooled_geos` and their results, and makes public the engines' fits and specs that public
functions take or hold; renames `stacked` and `lift_row_pick` by ADR 0062's removal rule; amends
ADR 0062: every module's `__all__` is public, and `tests/test_public_api.py` holds the surface to a
snapshot; changes no record, no pre-registration and no arm's name; amended 2026-11-25 by ADR 0103:
the snapshot at the last release tag holds the surface, and the check of the classes public names
take follows a `type` alias to its value.

## Context

Three pre-registered runs scored the engine's plans (`docs/benchmark.md`). Each verdict names the
arm that planned: `margin held` at k = 0, `anchored floored` at the tested rung, and `geo eb` on
the geo families. The verdicts that hold cover 37 family-rungs. None of these arms is a public
name. They are assembled in `scorecard/arms/`, which is outside the wheel. They read the bench's
exports, whose lift tests are lift rows: each test's mean weekly change and its standard error.
They compose functions that sit at module paths: `mixdyn.models.margin_mmm`,
`mixdyn.decisions.margin_plan`, `mixdyn.calibration.anchors`, `mixdyn.models.geo_eb` and
`mixdyn.decisions.geo_plan`. The `anchored` arm also reads the bench's `TEST` and `COOLDOWN`. A
caller who wants the plan the record describes has to rebuild an arm from those parts. The README's
first plan did that, and said that its names were not public yet.

The surface had three gaps besides:

- **Public functions took classes that were not public.** `readings`, `stacked`, `lift_row_pick`,
  `compare_scenarios`, `msm_sensitivity` and `transforms_from_bayes` take `BayesFit`. The others:
  - `factsheet` takes `BayesSpec`;
  - `geo_factsheet` takes `BayesGeoFit`;
  - `ridge_factsheet` and `ridge_sensitivity` take `RidgeFit`;
  - `decision` takes `HealthCard`;
  - `DlmFit.lift` holds `Lifted`;
  - `DemandProxy` holds the ridge model's `Control`.

  Those classes' own fields held seven more: `BayesMedia`, `Media`, `LinearTrend`,
  `FourierSeason`, `Calibration`, `Diagnostics` and `LiftReading`. ADR 0062 left the engines'
  entry points at module paths "until an `__all__` lists them".
- **ADR 0062 and its test named different surfaces.** The ADR listed the packages' `__all__`s and
  that of `mixdyn.api.serialize`. `tests/test_public_names.py` and `docs/api/index.md` also read
  the module-level `__all__`s of `mixdyn.data.feedback`, `mixdyn.models.demand` and
  `mixdyn.inference.sensitivity`.
- **Nothing held a public name's signature.** The index lists names, not parameters or fields,
  and the script that checks it also writes it. So a removed name passed once the page was written
  again, and the removal rule of ADR 0062 was held by review alone.

At the tested rung, the scored arm floors each free channel at 0.9 times its least weekly spend in
the history (ADR 0099). The paired difference from the same rule without the floor, `anchored held`,
is a euro of budget; a positive one is what the floor costs:

- **ADR 0099's final half** (2 140 fresh worlds): −0.0102 on family 6, +0.0294 on family 19,
  +0.0123 on family 4, +0.0038 on family 20, and +0.025 on family 15 (M3's records).
- **M4's and M5's readings of their recorded arms** (`evidence/results/floor_record.json`, written
  by `scripts/floor_record.py`), post hoc and unadjusted: +0.012 on family 4, −0.005 on family 6,
  +0.004 on family 14, +0.027 on family 19, +0.004 on family 20, and +0.026 on family 15 (M5).
  Every other family's is within ±0.003.

The floor gains only on family 6, whose truths fall off below the logged range. All 16
tested-rung verdicts rest on the floor; no run declared the rule without it.

## Decision

### The planners

`mixdyn.decisions` gains three planners, in `mixdyn.decisions.scored`. Each fits and plans in one
call, as an arm does, and has the shape of `plan_posterior`: the panel, the lift tests where the
rule reads them, the budget, the periods, the channels' `Range`s, and the kernel's length.

| Public name | Recorded as | What it reads |
|---|---|---|
| `plan_margins` → `MarginPlan` | `margin held` | One national weekly history; no lift test. |
| `plan_anchored_margins` → `MarginPlan` | `anchored floored`, and `anchored held` with `spend_floor=None` | One national weekly history, and at least one `LiftEstimate` for each channel. |
| `plan_pooled_geos` → `PooledGeoPlan` | `geo eb` | A weekly geo panel with populations, and the geos' `LiftEstimate`s, if any. |

- **A planner's procedure is its arm's.** The arms keep their code, which their pre-registrations
  froze, and their names. `tests/scorecard/test_public_planners.py` holds each planner to its arm,
  to the last bit of the plan, its gain and its interval: `plan_margins` on the six worlds of the
  margin rule's fixture, `plan_anchored_margins` with and without its floor on the eight of the
  tested rung's, and `plan_pooled_geos` on family 8's pilot world at both rungs.
- **Lift tests are `LiftEstimate`s.** A test's dark weeks and its readout are read off the estimate,
  not off the bench's constants: a go-dark of one channel over the whole market, ending on a date
  of the panel's. The reduction of an estimate to the arm's lift row is the one new numerical path.
  The same test holds it to the bench's lift rows on every fixture export, bit for bit.
- **`spend_floor: float | None = 0.9`**, a share in (0, 1] of each free channel's least weekly
  spend, or `None` for no floor. 0.9 is the default: the 16 tested-rung verdicts were scored with
  it. Only 0.9 was scored, and a share of 1.0 failed ADR 0099's no-loss bar. `plan_margins` takes
  no floor: it already cuts each box to the history's adstock range. A plan's `record` names the
  arm whose records it reproduces, and is None at a share no run recorded.
- **A planner refuses what its rule does not read.** It refuses a panel that is not weekly; the two
  national rules refuse a geo panel, a control other than `promotion` and `price`, a filled week and
  an event; `plan_anchored_margins` refuses a channel without a test. It accepts any budget and any
  box. Its docstring says that the record covers a budget equal to the status quo's weekly total,
  which a `budget` of None spends, and states the record's line: the family-rungs within δ, by run,
  and those not within it. `tests/test_readme.py` holds that line to the records.
- **The results are registered**, so `save` writes them: `MarginPlan`, `PooledGeoPlan`, and the
  fits they are planned from, `MarginFit` with its `NoiseMixture`, and `PooledGeoFit` with its
  `Mixture` and `CellResponse`. `GeoEbFit` became `PooledGeoFit`, and `fit_geo_eb` became
  `fit_pooled_geos`, before either entered `REGISTRY`.
- **Made public**, each in the module that defines it, so no `REGISTRY` key moves:
  - the fits and specs that public functions take or hold, and the classes their fields hold:
    `BayesSpec`, `BayesFit`, `BayesMedia`, `BayesGeoSpec`, `BayesGeoFit`, `RidgeSpec`, `RidgeFit`,
    `Media`, `Control`, `LinearTrend`, `FourierSeason`, `Calibration`, `NestedSpec`, `NestedFit`,
    `Lifted` and `LiftReading` in `mixdyn.models`, and `Diagnostics` and `HealthCard` in
    `mixdyn.inference`;
  - their `fit_*` functions: `fit_bayes`, `fit_bayes_geo`, `fit_ridge` and `fit_nested`;
  - the margin regression: `fit_margins`, `MarginFit`, `NoiseMixture` and `DegenerateFitError`;
  - the model over geos: `fit_pooled_geos`, `PooledGeoFit`, `Mixture` and `CellResponse`.

  The planners' internals stay at their module paths.

### The names the records hold

Arm names never change. They appear in the pre-registrations, whose SHA-256s the bench's signed
tags carry, in `evidence/results/`, and in the scorecard. `scripts/readme_record.py` maps each
claim arm to the public planner that reproduces it, and the benchmark page's claim table prints
both. No arm takes a public name. `gs`, `gs curved` and `gs held` have no planner: their route
caused M3's two losses past δ. The `margin` and `anchored` ablations stay recorded variants.

### Two renames

`stacked` becomes `stacked_readings`, because the bare name is the scorecard's arm. `lift_row_pick`
becomes `pick_by_lift_tests`, because a lift row is the bench's export, not a public type. Each old
name leaves `__all__` for the package's `_DEPRECATED` and imports through its `__getattr__`, with a
`DeprecationWarning` that names its replacement and the release it leaves in, 1.0.0, the release
after 0.3.

### The surface

- **Every module's `__all__` is public.** This amends ADR 0062's list of public modules to what its
  test already reads.
- **`tests/test_public_api.py` holds the surface to `tests/public_api.txt`**, one entry per public
  object: a function's signature; a class's decorators, bases, public fields, class attributes,
  enum members and methods; a constant's value; the package a name of another package comes from;
  and a line for each name being removed, with its replacement and its leaving release. The entries
  are read from the source's syntax tree, so the text is the same on Python 3.12 to 3.15. Its
  writer refuses to drop a name before the release its package's `_DEPRECATED` says it leaves in,
  so writing the snapshot again no longer hides a removal. The test also checks four rules:
  - each name in a `_DEPRECATED` is out of `__all__`, its replacement in it, and it imports with
    its warning until its release; from 1.0 it leaves only in a major release;
  - no public name shadows a submodule of its package, but the three that did before the rule
    (`mixdyn.decisions.plan_table`, `mixdyn.decisions.returns` and `mixdyn.inference.loo`), a list
    that only shrinks;
  - no public callable or class takes, returns or holds a mixdyn class that no `__all__` lists;
  - each planner's docstring names the arm it was recorded as, with that arm's record
    (`tests/test_readme.py`).
- **From 1.0**, a default that moves a plan changes only in a minor release, with an ADR and a
  *Changed* entry in the changelog. A planner's default changes only on a confirmatory run's
  record.

## Consequences

- The plan that the benchmark describes is one public call. The README's first plan calls
  `plan_margins`.
- The record's claim describes the defaults. With `spend_floor=None`, the plan is `anchored held`'s,
  which no run has claimed. The docstring says so.
- The ceiling the margin planners missed by a last bit (`plan + (high - plan)` is not `high`) is
  fixed in a commit of its own, before the equivalence test was pinned. It moves a plan by at most a
  unit in the last place, and no fixture's plan. Scored records are not read again: a run is read
  with the tree that ran it.
- To revisit the floor's default, a confirmatory run declares `spend_floor=None` at the tested rung.
  `Declaration.default_less` (ADR 0094) reads rung 0 alone, so either it is extended to the tested
  rung or the run claims both settings.
- 0.3.0 carries the additions, the renames with their warnings, and this amendment. 1.0 drops the
  two old names and freezes what the snapshot holds.

## Alternatives considered

- **Public names for the arms as they stand, `scorecard.arms` in the wheel.** Rejected: the arms
  read the bench's export and constants, and the bench is not a runtime package
  (`allowed-runtime-packages.txt`).
- **The arms planned through the public planners.** Rejected: an arm's code is what its
  pre-registration froze, and the bench's export is what it reads. The equivalence test holds the
  two to each other instead.
- **One planner that picks the rung from the data.** Rejected for now: it hides which record a plan
  rests on. It can be added later without a break; removing one later would be a break.
- **Drop the spend floor.** Rejected: the default would leave all 16 tested-rung verdicts, and the
  rule without the floor has only undeclared readings.
- **The floor fixed, with no parameter.** Rejected: it costs 0.004 to 0.029 a euro on five of 16
  families, and `None` reproduces `anchored held` exactly.
- **The planners' internals public as well** (`shrink_margins`, `water_fill`, `time_held`, the
  ranges, `anchors`). Rejected: their signatures follow the bench's export, and freezing them would
  freeze the arms' internals.
- **Renames without aliases, as SemVer allows before 1.0.** Rejected: ADR 0062's removal rule
  applies before 1.0 too.
