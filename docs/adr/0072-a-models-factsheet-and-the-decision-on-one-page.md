# 0072. A model's factsheet and the decision on one page

Status: accepted, 2026-11-06; amended 2026-11-07: the decision names the channels its plan moves on
their prior alone, at version 1 before the first release; the builders of the other engines (below)

## Context

A buyer of a media-mix model asks two things of it before the numbers: what the model rests on, and
what to do. The fields a model states here follow the Coalition for Innovative Media
Measurement's (CIMM) roadmap for media-mix models: its data's sources and transforms, its priors
and where they come from, each experiment it was calibrated to, a holdout, ranges of its returns,
the power of its tests, how often it is refreshed, and its limitations.

The engine returned each of them apart, as Python objects: a spec and its priors in a module's
docstring, the tests' estimands (`LIFT_TESTS`), the health card (ADR 0060), a plan (ADR 0019,
0071), a set of tests by its value (ADR 0031). Nothing put them on one page, and nothing said
when one was missing.

## Decision

- **Two documents of `mixdyn.api`, each a schema of its own**, beside `ISSUES` and `LIFT_TESTS`,
  whose versions do not move: `FACTSHEET = Schema("mixdyn.factsheet", 1, Factsheet)` and
  `DECISION = Schema("mixdyn.decision", 1, Decision)`. Each reads and writes as every document
  does (`dumps`, `loads`), field by field by its declared type, and each class is in the file
  format's registry (ADR 0062).
- **A field with nothing to record holds `Unfilled(reason)`**, never a blank, a `None` or a
  zero. The reason is not blank. A union of a record and `Unfilled` is written with its `kind`.
- **`factsheet(panel, geo, spec, returns, ...)`** states the one-geo Bayesian model:
  - `model`, `kpi`, `cadence`, the calendar's `first` and `last` dates, `geos` and `channels`;
  - `sources`: each column the model reads, its role, its origin as the caller states it (`"not
    stated by the caller"` otherwise) and the transforms the reader and the model run on it, in
    order: a step with no input row read as unobserved, a national channel's split over the geos,
    the scaling, the carryover and the curve, a control's standardisation and its sign;
  - `priors`: each prior the model states, with its provenance: a return's prior is the engine's
    default, Meridian's for a revenue KPI, only where it is `RoiPrior()`, and stated by the caller
    otherwise;
  - `tests`: each lift test as `estimands` reads it, its estimand and its weeks;
  - `returns`: each channel's mean and central interval over draws of its return, what the return
    counts named;
  - `holdout`, `power` and `refresh`, the caller's records. The engine runs none of them, so each
    is `Unfilled` with its reason by default; `power` (backtests and the minimum detectable
    effect's range) stays unfilled until a geo design's power audit fills it;
  - `limitations`: the caller's, then the engine's, which the fields imply: a channel no test
    calibrates, the confounding no column holds, a holdout or a power audit not recorded.
- **`decision(panel, geo, plan, *, title, ...)`** is a `PosteriorPlan` on one page:
  - the plan: its periods, budget, gain, the current spend's gain and each channel's current and
    planned spend;
  - `health`: the health card's cells, the thresholds the caller passed by `HealthCard.flags`'
    keywords, and the flags they read. No threshold is a default (ADR 0060);
  - `reconciliation` (`reconcile(dates, kpi, parts)`): each period's parts, their total summed in
    the parts' order, the KPI and the residual, the KPI less the total;
  - `next_test`: the set `choose_tests` chose, its verdict, its tests, their value, cost and the
    value of perfect information; unfilled, with that value, where no set is affordable;
  - `events`, the decision's progress, dated; and `compute`, the device and the wall-clock seconds
    as the caller recorded them, a record and not a claim.
- **`render(document)`** writes either as plain text: a field a line under its name, nested fields
  indented, an item of a tuple after a dash, an unfilled field with its reason, the
  reconciliation as a table a period a row. Every number is written at four significant digits at
  most, without an exponent.

## Consequences

- `tests/test_factsheet.py` generates four reports on a tiny world, twenty weeks of two channels
  and a control: a filled factsheet, an unfilled one, a filled decision and a plan alone. Between
  them each union field is both filled and unfilled. For each report every field is named in its
  class's docstring, is in its JSON document, is on its page in order under its name, and the
  report reads back from JSON and from a saved file equal to itself.
- The decision page of the tiny world is the golden file `tests/golden/decision.txt`, regenerated
  by `PYTHONPATH=src python tests/test_factsheet.py > tests/golden/decision.txt`. A change that
  moves a number at its fourth significant digit, a field's name or its order fails the test; one
  below it does not.
- The world's reconciliation holds its residual to the KPI's noise within `1e-9`; each total is
  its parts summed in order, exactly.
- The factsheet's priors are read off the spec as the model's density reads them: the slope's
  only under a Hill, `nu`'s only under Student-t noise, the controls' and the calendar's only where
  the model has them; a test holds each.
- A field added to either document is a new version of its schema, as `LIFT_TESTS`' would be.

## Alternatives considered

- **One document holding both**: the factsheet changes when the model or its data does, the
  decision each time a plan is made. Two schemas version apart.
- **`None` for a missing field**: a reader cannot tell a field the engine forgot from one with
  nothing to record. `Unfilled` names why, and the page shows it.
- **A golden JSON document**: its floats are written to the last bit, so a change of a JAX kernel
  in the last place would fail it. The rendered page rounds them; its JSON is checked by the round
  trip instead.
- **A power audit computed here**: the geo design's backtests and minimum detectable effects
  belong to the design (ADR 0055); the field waits for them.

## Amended 2026-11-07: the channels a plan moves on their prior alone

- `Decision` gains `unidentified`, the plan's `PosteriorPlan.unidentified` by name (ADR 0044): the
  channels it moves whose spend never moved in the geo. It is the last field, so the page writes it
  last, `unidentified: none` where the plan names none; it defaults to empty, so a decision saved
  to a file before it loads (ADR 0062).
- The schema stays `"mixdyn.decision"` version 1. The rule above, "a field added to either
  document is a new version of its schema", is ADR 0063's rule from the first release; before it a
  schema changes in place, and mixdyn has not been released. A version-1 document written without
  the field is refused, the field named, as reading is strict. From the first release such a field
  moves the version.
- `tests/test_factsheet.py`: a fifth report, a plan that names search, is on its page and in its
  document field by field; the golden page gains the line `unidentified: none`.

## Amended 2026-11-07: the factsheets of the model over geos, the DLM and the ridge

`factsheet` stated the one-geo Bayesian model alone; the other engines' factsheets were written by
hand. Each engine now has a builder that reads its own fit into `FACTSHEET`'s one schema, version 1
unchanged: no field is added, and a field an engine cannot fill says why where the schema lets it.

- **`geo_factsheet(panel, fit, ...)`**, the Bayesian model over geos (`BayesGeoFit`): the calendar
  is the fit's window and the geos every geo; the sources hold the steps the model runs over the
  geos, a geo's KPI and spend each divided by its population, a channel's carryover and curve
  shared by the geos and its effect pooled over them; the priors are the model's, the spread's
  and the bases' with them, the spread's Meridian's at 1; each return is the fit's draws of the
  channel's return over the window and every geo. A channel bought nationally and split by
  population is a limitation: the geos hold no contrast in its spend.
- **`dlm_factsheet(panel, geo, fit, decomposition, ...)`**, the DLM (`DlmFit`): the returns are read
  off the draws of `decompose_dlm`, which the caller passes, each channel's contribution over the
  fit's periods over its spend there, so a builder draws nothing of its own. The sources hold each
  effect's transform with its numbers, a lever's level, its coefficient's discount and sign. The
  priors are the state's and the variance's before the first period, `vague_prior`'s, its numbers
  and what they are made of, or the caller's, then each block's discount. The limitations add
  the transforms held at their values, coefficients that walk, and signed draws that have not
  mixed.
- **`ridge_factsheet(panel, geo, fit, ...)`**, the ridge (`RidgeFit`): the ridge is a point, so each
  return range is the point at a level of 0, its low and high the point, and a limitation says so.
  The ridge states no prior: its penalty, and each geo test's anchor where the fit read tests, are
  listed in the priors' place, each saying it is not a prior. Its geo tests (`chc.lift.LiftTest`)
  are not estimands `estimands` reads, so `tests` holds none and a limitation names the channels
  they read.

Consequences:

- `tests/test_factsheet_builders.py` builds each on a tiny world: three geos of 30 weeks with a
  national channel and a step with no row, fitted by 2 chains of 150 draws; 40 weeks of TV, search
  and a price lever fitted by a DLM with a signed, transformed TV and a lift test, and by the
  ridge; and the ridge calibrated by geo tests. Every field of each is on its page under its name,
  in its JSON document, and reads back the same from JSON and from a file. Each return range is
  the quantiles of its own draws, or the point, to the bit.
- The one-geo builder's output does not move: its limitations and checks now come from helpers
  the four builders share.

Alternatives considered:

- **An interval for the ridge's returns from a bootstrap.** Not here: it would be a new estimate,
  not a statement of the fit. A caller who bootstraps the ridge can state the draws through
  `factsheet`'s fields by hand.
- **A union `returns: tuple[ReturnRange, ...] | Unfilled`.** A new version of the schema, for a
  reading the point and its limitation already give.
- **The DLM's builder drawing its own decomposition.** A document builder that draws hides a seed;
  the caller's `decompose_dlm` names it.

