# The public API

This page is written by `scripts/api_index.py` from each package's `__all__` and each name's own
documentation, and `tests/test_public_names.py` fails when the engine and the page disagree. Each
name is listed by the packages that list it, with the first sentence of its docstring; its fields,
methods and the rest are in its docstring. Decisions the names rest on are in `docs/adr/`.

## What is public

- **A name is public when an `__all__` of a package lists it**, and so are the fields, methods
  and properties of the classes these name that do not start with an underscore (ADR 0062). A
  name reachable only by its module path is not public, however much it is used.
- **Every public name imports and is documented**: by a docstring of its own, the string after a
  constant or an alias in the module that defines it, or a page of `docs/` that cites it. This
  page cites every name and so documents none: `tests/test_public_names.py` lists the names not
  documented yet, and that list only shrinks.

## Versions and deprecation

- **Before 1.0**, a minor release may change the public surface, and only as the removal rule
  below allows; a patch release never removes or changes a public name. **From 1.0**, a name
  leaves only in a major release.
- **How a public name is removed:** for one minor release it still imports, and using it raises a
  `DeprecationWarning` that names its replacement and the release it leaves in; it leaves in the
  minor release after, and the changelog names it under *Removed*. causal-hybrid-control removes
  its names the same way.
- A renamed class that `save` writes stays in the format's registry under its old key for as long
  as its old name imports, so files that name it still load.

## The file format and the documents

- **`save` and `load`** (`mixdyn.api.serialize`) write and read one format, a zip of a JSON
  manifest and NumPy arrays; nothing is pickled, and a loaded object equals the saved one to the
  bit (ADR 0062).
- **`FORMAT_VERSION` is 1.** It moves when a file the current version wrote would load
  differently or not at all: a registered class or a field renamed, moved or removed, a field
  added without a default, or a field's meaning or units changed. Adding a class to the registry,
  or a field with a default, does not move it. `load` reads its own version only and refuses any
  other, naming both. The release that moves the version reads the previous one too, through a
  converter, for at least one minor release.
- **The JSON documents** (`dumps`, `loads` and their schemas) are for callers outside Python. Each
  names its schema and an integer version, and reading refuses a document of another version.
  ADR 0063 states when a schema's version moves.

## The names, by package

### `mixdyn`

- `aggregate`: The panel at `nodes` of its hierarchy, each node a geo of the result.
- `allot_national`: The panel with `channel` added, bought nationally: each period's national
  `spend` split over the geos `by` their population.
- `Allotment`: How a channel bought nationally is split over the geos, period by period.
- `Baseline`: A lever's counterfactual level read off the panel: the least or the most the lever was
  observed at, over every geo's periods that were not filled.
- `Cadence`: The spacing of consecutive periods.
- `Channel`
- `Combine`: How the controls of the geos below a node make the node's.
- `Count`: The KPI counts events (orders, installs, sign-ups) named by `unit`.
- `Gap`: What a period becomes when a geo has no row for it.
- `Geo`
- `Issue`: One finding of `read_panel`: what it is, how severe, and where.
- `IssueCode`: What a finding is about; its value is the code a document of `ISSUES` writes.
- `Lever`: A control the panel declares a lever: its `kind`, and the level its contribution is
  measured against, `baseline`, the lever's observed least or most or a level given in the control's
  own units.
- `LeverKind`: What a lever is: a column of the panel, not media spend, that moves the KPI.
- `MediaPanel`: A KPI, media spend and controls for every geo and period of a regular calendar.
- `NATIONAL`: The geo of a panel read without a geo column.
- `PanelError`: The data cannot be read as a panel.
- `read_panel`: Check one row per geo and period, then read it into a `MediaPanel`.
- `Revenue`: The KPI is revenue in the panel's currency: a return on spend is a ratio of one
  currency.
- `Severity`: Whether a finding stops reading.

### `mixdyn.api`

- `Compute`: The compute a decision took as it was recorded: the `device` it ran on and the
  `wall_seconds` the caller measured.
- `DECISION`: A plan on one page, with what it rests on: `decision`.
- `Decision`: A plan on one page, with what it rests on.
- `decision`: The decision document of `plan` (`plan_posterior`) for `geo` of `panel`, named
  `title`.
- `dlm_factsheet`: The factsheet of the dynamic linear model (`mixdyn.models.dlm`), `fit`, fitted to
  `geo` of `panel`, its returns read off `decomposition`, draws of the fit's decomposition
  (`decompose_dlm`).
- `dumps`
- `Estimand`: What one lift test of `channel` estimated, as a model reads it.
- `estimands`: Each test of `estimates`, each channel's in the order given, as the `Estimand` it
  reads on `panel`'s calendar, cadence and currency.
- `Event`: A step of the decision's progress: `what` happened `on` a date.
- `FACTSHEET`: What a fitted model is and what it rests on: `factsheet`.
- `Factsheet`: What a fitted media-mix model is and what it rests on, in fields that follow the
  Coalition for Innovative Media Measurement's (CIMM) roadmap for media-mix models.
- `factsheet`: The factsheet of the one-geo Bayesian model `spec` (`mixdyn.models.bayes_mmm`) fitted
  to `geo` of `panel`.
- `geo_factsheet`: The factsheet of the Bayesian model over geos (`mixdyn.models.bayes_geo`), `fit`,
  fitted to every geo of `panel`, read off the fit as `factsheet` reads the one-geo model off its
  spec.
- `Health`: The health checks of a fit (`mixdyn.inference.diagnostics.HealthCard`), each read
  against the threshold the caller passed: `max_rhat`, `ppp` and its Monte Carlo error `ppp_se`,
  `negative_baseline`, and each channel's `shift`, `width` and `consistency`, as the card holds
  them; `thresholds` the thresholds passed, by `HealthCard.flags`' keywords, and `flags` whether
  each cell passes, as `HealthCard.flags` reads it.
- `Holdout`: The periods from `first` to `last`, `periods` of them, held out of the fit, and how far
  its forecast of the KPI missed them: `error` by the `metric` named.
- `ISSUES`: What reading a panel found: `MediaPanel.issues`, or `PanelError.issues`.
- `LIFT_TESTS`: What each lift test a model reads estimated: `estimands`.
- `loads`
- `NextTest`: The next set of lift tests by its value to the plan (`ExperimentChoice`): the
  `verdict`, to `"test"` or to `"plan now"`; the set's `tests` by their labels and their `channels`;
  the regret they are expected to remove, `reduction`, and its Monte Carlo `error`, in the KPI's
  units; their `cost`, `net_value` and `predicted_net_value`, in revenue; and `evpi`, the value of
  perfect information, with its `evpi_error`, over `draws` outer draws.
- `Period`: One period of a reconciliation: its `date`, the `kpi` observed, its `parts`, each part's
  contribution in the order `Reconciliation` names them, their `total` and the `residual`, the KPI
  less the total.
- `PowerSummary`: How the tests' design was audited before they ran: `backtests`, the number of
  placebo windows of the history the design was read on, and `mde_low` to `mde_high`, the range of
  its minimum detectable effect over them, in the KPI's units.
- `Prior`: A prior the model states: the `parameter` it is on, its `distribution` on the model's
  scales, and its `provenance`, where its numbers come from.
- `reconcile`: The table that sums `parts`, each part's contribution to the KPI at each of `dates`,
  to a total in the parts' order, beside the `kpi` observed and its residual, the KPI less the
  total.
- `Reconciliation`: The contributions of `parts` to the KPI over the fitted `periods`, which sum to
  each period's total; the KPI is that total and the residual.
- `Refresh`: How often the model is fitted again: every `every` periods of `cadence`, last on
  `last`.
- `render`: `document` as plain text: its kind on the first line, then each field on a line of its
  own under its name, a nested one indented below it, an item of a tuple after a dash, an `Unfilled`
  one as `unfilled:` and its reason, and a reconciliation's periods as a table, a period a row.
- `ReturnRange`: A `channel`'s return over a posterior's draws: their `mean` and the central `level`
  interval from `low` to `high`, in the KPI's units a currency unit, and `counts`, what the return
  counts (`RoiPrior.counts`): the window's contribution over its spend, or the window's spend's
  whole return with its carryover.
- `ridge_factsheet`: The factsheet of the ridge model (`mixdyn.models.ridge_mmm`), `fit`, fitted to
  `geo` of `panel`.
- `Schema`: A document type: its name, its version and the declared type of its data.
- `SchemaError`: A value or document does not fit its schema; the message names where.
- `sensitivity_line`: ADR 0061's line on a channel's sensitivity to an unmeasured confounder: the
  ridge fit's robustness values and, where `control` is named, its coefficient under a confounder as
  strong as that control; the Bayesian fit's `Gamma*` and its bound at `gamma`.
- `Source`: A column the model reads: its `column` name, its `role` (`"kpi"`, `"spend"`, `"control"`
  or `"calendar"`), its `origin`, where its data came from as the caller states it, and the
  `transforms` the reader and the model ran on it before the likelihood read it, in order.
- `Spend`: A `channel`'s spend a period, in the panel's currency: the `current` and the `planned`.
- `Unfilled`: A field of a document with nothing to record, and the `reason` it has none.

### `mixdyn.api.serialize`

- `FORMAT`: The name a file's manifest gives its format.
- `FORMAT_VERSION`: The version of the format this mixdyn writes, and the one version it reads.
- `FormatError`: A file that is not one this mixdyn reads: not an archive of the format, of another
  format version, or naming a class or a value the format does not hold.
- `load`: The object `save` wrote to `path`.
- `REGISTRY`: Every class a file may hold, by `module:qualname`.
- `save`: Write `obj`, an instance of a class in `REGISTRY`, to `path` as the format's archive.

### `mixdyn.calibration`

- `Estimates`: Lift estimates by channel, as the models take them.
- `experiment_table`: Lift estimates as experiments' records, the columns `read_experiments` reads
  with its estimate a lift: each estimate's lift and standard error, the window its spend changed
  over by its first and last dates, the periods read after it, its spend and planned spend a period
  over the window, its geo where it ran in one and its share of the market.
- `lift_table`: Lift estimates as PyMC-Marketing's lift table, the columns `read_lift_table` reads:
  each estimate's lift a changed period and its standard error a changed period, its spend before
  the test and the change.
- `LiftEstimate`: A lift test of one channel as its analysis reported it.
- `LiftSeries`: A lift test of one channel read period by period: `LiftEstimate`'s test with its
  lift over each period of the readout rather than summed.
- `read_experiments`: Geo experiments' records as lift estimates on `panel`: each record a test of
  `channel` whose group spent `spend` a period against `planned` over its window, the periods from
  `start` to `end` on the panel's calendar, its lift read over the window and `cooldown` periods
  after it at `planned`, none where the column is absent.
- `read_lift_table`: PyMC-Marketing's lift table as lift estimates on `panel`: each row a test that
  changed the spend from `x` to `x + delta_x` a period for `periods` periods, its lift read over
  those and `cooldown` periods after at `x`, `delta_y` times `periods` and its standard error
  `sigma` times `periods`.

### `mixdyn.data`

- `aggregate`: The panel at `nodes` of its hierarchy, each node a geo of the result.
- `allot_national`: The panel with `channel` added, bought nationally: each period's national
  `spend` split over the geos `by` their population.
- `Allotment`: How a channel bought nationally is split over the geos, period by period.
- `Baseline`: A lever's counterfactual level read off the panel: the least or the most the lever was
  observed at, over every geo's periods that were not filled.
- `Cadence`: The spacing of consecutive periods.
- `Channel`
- `ColumnData`
- `Combine`: How the controls of the geos below a node make the node's.
- `Count`: The KPI counts events (orders, installs, sign-ups) named by `unit`.
- `Gap`: What a period becomes when a geo has no row for it.
- `Geo`
- `Issue`: One finding of `read_panel`: what it is, how severe, and where.
- `IssueCode`: What a finding is about; its value is the code a document of `ISSUES` writes.
- `KpiUnit`
- `Lever`: A control the panel declares a lever: its `kind`, and the level its contribution is
  measured against, `baseline`, the lever's observed least or most or a level given in the control's
  own units.
- `LeverKind`: What a lever is: a column of the panel, not media spend, that moves the KPI.
- `MediaPanel`: A KPI, media spend and controls for every geo and period of a regular calendar.
- `NATIONAL`: The geo of a panel read without a geo column.
- `PanelError`: The data cannot be read as a panel.
- `read_panel`: Check one row per geo and period, then read it into a `MediaPanel`.
- `recurring_events`: The events that recur in `panel`'s weeks of the year, read off its controls
  `indicators`, each with its direction, 1 where an event raises it and -1 where it lowers it; see
  the module.
- `RecurringEvents`: The events `recurring_events` reads: each one's `weeks` of the year, a run of
  consecutive weeks from 1 to 52, named in `names`; each week of the year's `posterior` probability
  of the event class, `(52,)`, the prior's where the panel holds no week of it; and `separation`,
  the one class's Bayesian information criterion less the two classes', above 0 where two read
  better and the events are read.
- `Revenue`: The KPI is revenue in the panel's currency: a return on spend is a ratio of one
  currency.
- `Severity`: Whether a finding stops reading.

### `mixdyn.data.feedback`

- `spend_feedback`: Each of `geo`'s channels whose spend varies, read as the module states: `gamma`
  and its interval at `level`, 0.8, 0.9, 0.95 or 0.98, the levels whose fixed-b critical values are
  tabled, and the season `harmonics` harmonics of a year at the panel's cadence.
- `SpendFeedback`: How far `channel`'s spend in `geo` reads the KPI's last period.

### `mixdyn.decisions`

- `Breakdown`: How far a plan's effects may be off before its gain over a reference is gone.
- `breakdown`: The breakdown factor of each row of channels' `gains` over a reference, `(...,
  channels)`: the common factor `r*` by which effects must be off, overstated where a channel gains
  and understated where it loses, for the gains to sum to 0, `sqrt(sum g^+ / sum g^-)`.
- `compare_scenarios`: Each of `scenarios` over the `horizon` periods after the fit's window, read
  against the one named `baseline` at `draws` of the posterior and the noise drawn from `seed`, both
  shared by every scenario.
- `Costs`: One channel's costs per incremental KPI over a window: `spend` in currency;
  `incremental`, the KPI the window's spend earns wherever it falls, its carryover counted, `roi` in
  the KPI's units times the spend; `cost_per_kpi`, CPIK, the spend over that KPI; and
  `marginal_cost_per_kpi`, the cost of one more incremental KPI at the window's spend.
- `costs`: `channel`'s costs per incremental KPI over `window` of its `(T,)` spend series, nothing
  spent before it: one over `returns`' `roi` and `marginal_roi` in the KPI's units.
- `CvarPlan`: A plan of one geo's channels for the most mean gain over a reference in the worst
  `level` share of a posterior's draws, and that reference.
- `FlightPlan`: A schedule of one geo's channels over periods, the best rate a period beside it, and
  how each and the status quo score over a posterior's draws.
- `GeoLimits`: What a plan may spend, by name.
- `GeoPlan`: A plan of every geo's channels, and the current spend it changes.
- `Goal`
- `lift_row_pick`: Each fit's weight when the lift rows alone choose among `fits`: 1 on the fit
  whose leave-one-out log predictive density, by PSIS-LOO, summed over the observations past the
  KPI's, the geo tests' gaps, the lift estimates and the lift series, is the largest, the first on a
  tie, and 0 on the others; None where the fits read no such observation.
- `MarginalReturnTarget`: The budget at which one more currency unit returns ``per_unit``: below it
  a unit returns more, past it less.
- `MarginPlan`: A plan of one national history's channels by a margin rule (`plan_margins`,
  `plan_anchored_margins`), and the status quo it moves.
- `media_difference`: What `plan_b` adds to the KPI over `plan_a` at each draw, `(draws,)`, read
  through the channels alone: each draw's channel run over its `history` and each plan's `periods`
  periods of spend and its kernel's reach after them, `length - 1` periods of no spend, and the
  difference summed over the plan's periods and the reach.
- `move_between_geos`: `plan`, each geo's spend by channel, with `amount` of `channel`'s spend moved
  from `from_geo`'s cell to `to_geo`'s, at each period where the cells are series; every other cell
  as it was.
- `plan_anchored_margins`: The plan of the panel's channels over `periods` weeks that spends
  `budget` over them within `limits`, each channel's range a week, by the margin rule anchored on
  each channel's lift tests, `estimates` by channel.
- `plan_breakdown`: The breakdown factor of `spend`, a plan of `geo`'s channels a period over
  `periods` periods of the panel's cadence, over `against`, or else each channel's mean spend over
  the geo's last `periods` observed periods as `plan_posterior` reads the current spend: for the
  posterior's expected gain and for each draw's, the history's carryover counted in.
- `plan_cvar`: The plan of `geo`'s channels over `periods` periods of the panel's cadence that
  spends `budget` over them within `limits`, each channel's range a period, for the most mean gain
  over `against`, the plan in place at this budget, in the worst `level` share of `draws` of the
  readings' draws: the gain's conditional value at risk, the history's carryover counted in.
- `plan_flighting`: The schedule of `geo`'s channels over `periods` periods of the panel's cadence
  that spends `budget` over them within `limits`, each channel's range in each period, planned on
  the readings' channels at `reading`, and the best rate a period beside it; each scored, with the
  status quo, at every draw.
- `plan_geos`: The joint plan of every geo's channels over `periods` periods of the panel's cadence
  that spends `target`, a budget over the plan, or meets it, a goal in its place.
- `plan_margins`: The plan of the panel's channels over `periods` weeks that spends `budget` over
  them within `limits`, each channel's range a week, by the margin rule with no lift test.
- `plan_pooled_geos`: The plan of the panel's cells over `periods` weeks that spends `budget` over
  them, each cell within its range a week in `limits`, by the closed-form model over geos fitted
  with the geo tests `estimates`, by channel, if any.
- `plan_posterior`: The plan of `geo`'s channels over `periods` periods of the panel's cadence that
  spends `budget` over them within `limits`, each channel's range a period, for the most expected
  worth over the readings' draws, each channel's mean worth read at `grid` spends evenly across its
  range.
- `plan_robust`: The plan of `geo`'s channels over `periods` periods of the panel's cadence that
  spends `budget` over them within `limits` for the most expected gain over `against` when each
  channel's effect may be off by a factor in `[1/scale, scale]`, its least over the factors:
  `plan_posterior`'s search on each channel's mean change in worth over the reference's, `x`,
  counted as `x / scale` above 0 and `scale x` below.
- `plan_table`: The table of `spend`, a plan of `geo`'s channels a period over `periods` periods of
  the panel's cadence, against `against`, or else each channel's mean spend over the geo's last
  `periods` observed periods, as `plan_breakdown` reads the current spend.
- `PlanTable`: A plan of one geo's channels against the current spend, one row a channel.
- `PooledGeoPlan`: A plan of a geo panel's cells by the model over geos (`plan_pooled_geos`), and
  the status quo it moves.
- `portfolio_roi`: Several channels' revenue over their spend: each channel's `roi` weighted by its
  spend.
- `PosteriorPlan`: A plan of one geo's channels for a posterior's expected return, and the current
  spend it changes.
- `Range`: At least `least` and at most `most`, in the panel's currency; `most` may be infinite.
- `ranges_from_current`: Each channel's range a period as a planner states it: at least
  `least_share` of its `current` spend a period, or its floor in `floors` where it has one, and at
  most `most_multiple` times its current spend.
- `Readings`: A posterior's draws as a planner reads them, each draw weighted.
- `readings`: `fit`'s posterior as a planner reads it: `draws` of its draws, evenly spaced over its
  chains' draws in order from the first to the last, or every draw, each weighted alike.
- `Recommendation`
- `response_curves`: Each channel's posterior mean worth and its central `level` band at each spend
  a period of `worths`, the draws at `readings.weights`: the band's ends are the weighted quantiles,
  each the least worth whose cumulative weight reaches its share, which at equal weights is
  `numpy.quantile`'s `inverted_cdf`.
- `ResponseCurves`: Each channel's worth at each spend a period, `(channels, grid)`, the channels in
  the readings' order: `rates`, the spends; `mean`, the posterior mean worth over the draws at their
  weights; and `low` and `high`, the weighted central `level` band.
- `ReturnOnSpendTarget`: The most budget whose plan gains ``per_unit`` for each currency unit it
  spends: a target return on ad spend, read on the gain.
- `Returns`: One channel's returns over a window: `spend` in currency, `contribution` in revenue,
  the returns in revenue per currency unit and `carryover` a share.
- `returns`: `channel`'s returns over `window` of its `(T,)` spend series, nothing spent before it.
- `ReturnTarget`: The least budget whose plan gains ``amount``, in the channels' units.
- `RobustPlan`: A plan of one geo's channels for the least expected gain over a reference when each
  channel's effect may be off by a factor in `[1/scale, scale]`, and that reference.
- `Scenario`: A named spend path over a forecast's horizon: `spend`, each channel's spend at each
  period, and `controls`, each control's value at each, a control it does not name held at the
  baseline's.
- `ScenarioComparison`: Scenarios read against a baseline at the same draws, in the panel's units.
- `Spread`: A quantity over a posterior's draws: `draws`, `(draws, ...)`, each draw's value; `mean`
  their mean by the draws' weights; `low` and `high` their weighted quantiles at `(1 - level) / 2`
  and `(1 + level) / 2`, each `(...)`.
- `stacked`: The fits' stacked posterior as a planner reads it: each fit's weight, `weights` or else
  stacking's on the fits' leave-one-out log predictive densities of the observations their
  likelihoods read, spread evenly over `draws` of its draws, evenly spaced over its chains, or over
  every draw.
- `Undefined`: A cost with no value, and why.
- `Worths`: Each draw's worth of each channel at spends a period across its range: `rates`,
  `(channels, grid)`, the spends, and `table`, `(draws, channels, grid)`, the worths, the channels
  in the readings' order.
- `worths`: Each draw's worth of each channel of `readings` over `periods` periods of the panel's
  cadence and its kernel's length after them, at `grid` spends a period evenly across the channel's
  range in `limits`, the geo's spend in the panel carried in.

### `mixdyn.experiments`

- `After`: A test run after the panel, its periods at `current`, each channel's spend a period under
  the plan in place, but for the test's change.
- `assign`: A design treating `share` of each block's geos, rounded half up, one of a pair and three
  of a block of five at the default, drawn by `seed`: the same seed draws the same design.
- `audit_power`: The power of `design`'s test of `readout` periods after `pre`, at each of `lifts`,
  read by `read` once over every backtest `kpi_history`, `(geos, periods)`, holds, by default with
  `read_did` on the groups' sums, at a 5 % level; see the module.
- `Balance`: Each feature's standardised difference between the treated and the control geos,
  `differences`, and the `largest` in absolute value.
- `balance`: Each of `features`' standardised difference over `design`: the treated geos' mean less
  the control geos', over `sqrt((s_t^2 + s_c^2) / 2)`, `s` a group's standard deviation.
- `Blocks`: Geos matched into `blocks`, each a tuple of rows in the order of the score, and the rows
  `left_out` of every block.
- `Candidate`: A lift test of `channel` before it runs: a `LiftEstimate` without its lift, and a
  `label` to name it by.
- `carryover_order`: The two-epoch test that `design`'s channel moves `kpi` no further than `order`
  periods after a switch, `order` the design's washout by default, `kpi` a value a period of the
  design or a row a geo of them, summed over the geos; see the module.
- `CarryoverOrder`: A switchback test's two-epoch test of its carryover's order: whether the channel
  still moves the KPI `order` periods after a switch.
- `choose_tests`: The set of `candidates` to run together, in disjoint groups of at most half the
  market between them, whose cost is at most `test_budget` and whose net value is the largest,
  valued as `value_of_tests` values one test, its tests' likelihoods multiplied: each candidate's
  lift read with its own standard normals, the same in every set it is in, drawn from a stream apart
  from `seed`'s own.
- `Cost`
- `designs`: Each of `channels`' candidates on `geo` of `panel`, in the order of the channels, the
  multipliers, the durations and the shares: the channel's spend times the multiplier over the
  duration, then `cooldown` periods at the spend planned, both read, in a group of the share, its
  standard error `noise`'s, run as `when` says.
- `ExperimentChoice`: The affordable set of tests of the largest net value, and whether to run it.
- `ExperimentValue`: The value to the plan of `candidates` run together, in disjoint groups, and its
  cost; one candidate's where `value_of_tests` reads it.
- `ExperimentValues`: Each candidate's value, in the candidates' order, and the plan without a test:
  `plan`, its spend a period in the readings' channels' order.
- `Gaussian`: Each outer draw's posterior mean worths moved by the Bayes-linear update: each fit's
  by the covariance of its draws' worths with the lift over the lift's predictive variance.
- `GeoDesign`: The rows of the `treated` geos and of their `control` geos.
- `GeoReading`: A geo test's reading by `method`: the `lift` over the readout and its standard error
  `se`, read as a Student t of `df` degrees of freedom; the two-sided interval `[lower, upper]` at
  level `alpha`; the `p_value` of the same t, two-sided, against no lift; and the `verdict`, "up"
  when the interval lies above 0, "down" when below and "inconclusive" when it holds 0.
- `GeoTest`: A geo test of `channel` as it ran: in the `treated` geos against the `control` geos,
  the spend changed from the period dated `start` to the one dated `end`, both included, and read
  over the `readout` periods from `start`.
- `Gross`: A test's cost as what it puts at risk: a heavy-up's extra spend and the revenue a cut is
  expected to lose, each counted where it is positive.
- `GroupNoise`: The standard deviation of the market's KPI noise a period, as it reaches a group's
  lift.
- `log_did`: The pre/post log-ratio difference in differences of a window and its standard error
  from placebo windows of the pre-period; see the module.
- `match_controls`: Each of `treated`'s control among `candidates`, one to one, minimising the
  summed Mahalanobis distance over `features`, each a value a geo, under the treated and the
  candidate geos' pooled within-group covariance.
- `Net`: A test's cost as its spend change plus the revenue it is expected to lose, held at least at
  `floor`: a cut of a channel returning less than it costs pays for itself, and a cost not above 0
  ranks nothing.
- `Noise`
- `pair_geos`: The geos matched into blocks of `size` adjacent in the order of `score`, one number a
  geo, minimising the summed range of the blocks; the `len(score) % size` geos left out are those
  that minimise it, the later in the order on a tie.
- `PlaceboNoise`: The market's KPI noise a period over the history before a test, `residuals`, the
  error a model of the KPI leaves in each period, and the `pre` periods each placebo window and the
  test are studentised over; see the module.
- `Power`: The power of a design's test at each of `lifts`, `power`, its `size`, the share of
  replications that read a lift where there is none, the minimum detectable lift at 80 % `mde`, NaN
  where no lift reaches it, and the count of `windows` the history holds.
- `power`: The power of `design`'s test of `readout` periods after `pre` at each of `lifts`, by
  `reps` replications drawn from `kpi_history`, `(geos, periods)`, by `seed`, each read by `read`,
  by default `read_did` on the groups' sums; see the module.
- `PowerAudit`: A design's power over every one of its `backtests`, at a 5 % level: its `power` at
  each of `lifts`; its `size`, the share of the backtests that read a lift where there is none; the
  minimum detectable lift at 80 % over them all, `mde`; and the least and the largest of it read
  over the first half of the backtests in time and over the second, `mde_low` and `mde_high`.
- `predicted_lift`: Each draw's lift of `candidate` at the market's scale, `(draws,)`, in the KPI's
  units: the channel's coefficient times the gap in its curve over the readout between the spend as
  the test runs and as planned, each over the share and carried over by the draw's kernel, nothing
  spent before the candidate's periods.
- `read_did`: `design`'s test on `panel` read by a difference in differences, its standard error
  from placebo windows with at least `warmup` periods before each, or, at `variance="unit_placebo"`,
  read by unit placebos over the design's matched `pairs`, drawn by `seed` where they are too many
  to enumerate; see the module.
- `read_pairs`: `design`'s test on `panel` read by its matched `pairs`' own difference in
  differences and sign-flip randomisation inference, the signs drawn by `seed` where they are too
  many to enumerate; see the module.
- `read_sdid`: A geo test in `panel` read by synthetic difference in differences: the geos in
  `treated` treated from the date `start` to the date `end`, both read, the geos in `control` the
  controls (every other geo where it is None) and every period before `start` the pre-period.
- `read_switchback`: `design`'s test read from `kpi` and the channel's `spend` as run, each a value
  a period of the design or a row a geo of them, summed over the geos; see the module.
- `read_synthetic_control`: `design`'s test on `panel` read by a synthetic control of the treated
  geos' mean, its weights fitted on the pre-period less its last `blank` periods, its standard error
  from the gaps' innovations over those `blank` periods; see the module.
- `read_tbr`: `design`'s test on `panel` read by Kerman, Wang and Vaver's (2017) time-based
  regression, its standard error from placebo windows with at least `warmup` periods before each, at
  `variance="iid"` their predictive variance under independent errors, or, at
  `variance="unit_placebo"`, read by unit placebos over the design's matched `pairs`, drawn by
  `seed` where they are too many to enumerate; see the module.
- `Reader`: A test's reading of its window: the treated and the control geos' KPI, `(geos,
  periods)`, the first `pre` periods before the test and the rest its readout, to the effect read
  and its standard error, the effect 0 where the test changed nothing.
- `Refit`: Each set of tests valued by `then` at every outer draw, and its posterior refitted with
  the tests' simulated estimates at `draws` outer draws of each half, evenly spaced: the mean gap
  between the refitted reductions and `then`'s there corrects `then`'s, a multi-fidelity estimate.
- `Reweight`: Each outer draw's posterior read by weighing the other chains' draws by the test's
  likelihood, and as `Gaussian` where the weights' effective sample size falls below `ess`.
- `SdidFit`: Synthetic difference in differences on one panel.
- `SdidReading`: A geo test as synthetic difference in differences read it.
- `seesaw`: Each of `channels`' seesaw candidates on `geo` of `panel`, in the order of the channels
  and the shares: the channel's spend times `high` and `low` in turn, `block` periods each, over
  `blocks` blocks from a high one, then `cooldown` periods at the spend planned, all read, in a
  group of the share, its standard error `noise`'s, run as `when` says; see the module.
- `stratified_design`: The comparator design over the geos of `kpi_history`, `(geos, periods)`, for
  a test of `readout` periods read after the history's last `pre`: strata of `size` geos alike in
  their series' mean, coefficient of variation and slope, `candidates` designs drawn within them by
  `seed`, each treating `share` of a stratum, filtered by the time-based regression's out-of-sample
  R^2 and an A/A test and the least minimum detectable lift chosen; see the module.
- `StratifiedDesign`: The comparator design `stratified_design` chose: its `design`, the `strata` it
  was drawn within, the count of distinct `candidates` drawn and of those `kept` by the R^2 and the
  A/A filters, and the chosen one's out-of-sample `r2` and its time-based regression's minimum
  detectable lift by formula, `mde`, a share of the treated geos' KPI.
- `switchback_design`: A switchback test over the periods dated `periods`, in blocks of `block`
  periods, the channel on in `round_half_up(share * blocks)` of them, in an order drawn by `seed`
  from `numpy.random.SeedSequence(seed).spawn(1)[0]`: the same seed draws the same design.
- `SwitchbackDesign`: A switchback test over the periods dated `dates`, cut into blocks of `block`
  periods, the channel on in the blocks where `on` is true; the first `washout` periods of every
  block after a switch, and of the first block, are left out of the reading.
- `SwitchbackReading`: A switchback test's reading by `method`: the `lift` over the periods on and
  its standard error `se`, for "randomization" the standard deviation of the lift over the
  arrangements, for "hac" Newey and West's and for "jackknife" half the gap between its halves'
  lifts, read as a Student t of `df` degrees of freedom (infinite for "randomization", which reads
  no t); the interval `[lower, upper]` at `level`; the `p_value` against no lift; the `verdict`,
  "up" when the interval lies above 0, "down" when below and "inconclusive" when it holds 0; the
  count of `arrangements` read, 0 for "hac" and "jackknife", and whether they were all `enumerated`;
  and the `carryover` check, NaN where no block after a switch keeps two periods in each state.
- `synthetic_did`: Synthetic difference in differences on `y`, units by periods, the units flagged
  in `treated` treated over the last `post` periods.
- `Update`
- `value_of_tests`: Each candidate's value to the plan of `geo`'s channels that spends `budget` over
  `periods` periods within `limits`, over `draws` outer draws of `readings`, each candidate's lift
  read with the same standard normals, drawn from a stream apart from `seed`'s own, and each outer
  draw's posterior updated by `update`; each channel's mean worth is read at `grid` spends across
  its range.
- `Verdict`
- `When`
- `Within`: A test run inside the panel's history from its period `start`, counted from 0, at the
  market's spend there but for the test's change.

### `mixdyn.inference`

- `Chains`: Each chain's draws after warm-up, ``(chains, draws)`` leading, with what each transition
  did, and each chain's step size and inverse metric as warm-up left them.
- `Diagnostics`: Per quantity, ``rhat``, ``ess_bulk`` and ``ess_tail``.
- `HealthCard`: The health of a Bayesian fit of one geo (`mixdyn.models.bayes_mmm`), each cell its
  number.
- `LogDensity`: A log density on the unconstrained space, up to a constant, and its gradient.
- `Loo`: ``pointwise`` is each observation's log predictive density left out, ``elpd`` their sum and
  ``se`` its standard error; ``p_loo`` the log predictive density in sample less ``elpd``, the
  effective number of parameters; ``pareto_k`` each observation's fitted shape, and ``k_threshold``
  the shape above which its estimate is not to be trusted.
- `loo`: ``log_likelihood`` is each draw's log density of each observation, ``(chains, draws,
  observations)``.
- `Psis`: Pareto-smoothed importance weights: ``log_weights`` normalised, ``pareto_k`` the fitted
  shape of the ratios' tail, infinite where there is none to fit, ``k_threshold`` the shape above
  which the weights are not to be trusted, and ``ess`` Kish's effective sample size of the weights,
  ``1 / sum w^2``, times the draws' relative efficiency.
- `psis`: Pareto-smoothed importance sampling of ``(S,)`` log importance ratios; ``efficiency`` is
  the draws' relative efficiency ``r_eff``, which sets the tail's length and scales the effective
  sample size.
- `sample`: NUTS chains from ``initial``, one row a chain, each adapting its step size and diagonal
  metric over ``warmup`` transitions and keeping the ``draws`` after; chain ``c`` draws its random
  numbers from ``numpy.random.default_rng([seed, c + 1])``, so that no chain reads
  ``default_rng(seed)``'s stream, which a caller may draw the starts from.
- `stacking`: The stacking weights of models, ``pointwise`` each one's log predictive density of
  each observation left out, ``(models, observations)``.

### `mixdyn.inference.sensitivity`

- `Benchmark`: An observed control's partial R^2: with the channel given the rest of the design,
  `r2_treatment`, and with the KPI given the channel and the rest, `r2_outcome`.
- `BiasBound`: A confounder's reach on a coefficient: it explains `r2_outcome` of the KPI's residual
  variance given the channel and the design, and `r2_treatment` of the channel's given the design.
- `msm_sensitivity`: The marginal sensitivity model's bounds on `channel`'s return in `posterior` at
  each of `gammas`, its intervals at `level`, and `Gamma*`.
- `MsmSensitivity`: A Bayesian fit's channel under the marginal sensitivity model.
- `ridge_sensitivity`: Cinelli and Hazlett's sensitivity of `channel`'s coefficient in `fit`, its
  intervals at `level`, read on the least squares of the fitted periods' KPI on the fit's design; a
  column that does not vary over them is left out with the intercept.
- `RidgeSensitivity`: A ridge fit's channel under Cinelli and Hazlett's sensitivity analysis.
- `robustness_value`: Cinelli and Hazlett's `RV_q` of a coefficient of t-statistic `t` on `df`
  residual degrees of freedom, or `RV_{q, alpha}` where `alpha` is given: the share of both residual
  variances a confounder must explain to move the estimate, or its `1 - alpha` interval's near end,
  by `q` of the estimate.

### `mixdyn.models`

- `BayesFit`: The posterior of the model fitted to one geo over a window of the panel's calendar.
- `BayesGeoFit`: The posterior of the model fitted to every geo of a panel over a window of its
  calendar.
- `BayesGeoSpec`: The model over geos: the one-geo model's channels, controls, calendar, noise and
  curves; the scale of the half-normal prior on each channel's log-SD of its effect over the geos;
  the channels whose deviations are `centred`, sampled on the scale of that log-SD; and three opt-in
  choices, the noise scale's prior, `"half-normal"` or `"log-normal"`, the geos' bases',
  `"log-normal"`, above 0, or `"normal"`, and a Hill's slope, `"sampled"` or `"fixed"` at 1.
- `BayesMedia`: A channel of the panel and the prior on its return.
- `BayesSpec`: The model: its channels, its controls, its trend and season if any, its noise, the
  family every channel's curve is of, and its kernels' length in periods, `kernel`: by default the
  longest series the model reads, the panel's calendar or a test's, and otherwise the periods
  stated, the weights normalised over them and a spend's carryover cut after them (ADR 0082).
- `Bounds`: An interval's ends.
- `Calibration`: How geo tests entered a fit: `estimates`, each tested channel's coefficient as its
  tests alone read it, and `weight`, a test period's weight against a KPI period, the KPI's noise
  variance over the tests'.
- `CellResponse`: A cell's expected response to its weekly spend over a plan: a mixture of curves,
  one a kept grid point, each of its `weights`, `retentions`, `curvatures` and `coefficients`.
- `CheckCode`
- `choose_dlm`: Filter `spec` on one geo at every discount of `grid` and choose by one-step
  likelihood; fit the specs a likelihood-ratio test at `level` does not reject, for
  `DlmChoice.bounds`.
- `ComponentBounds`: One interval per component, as `Components` holds one array of draws per
  component.
- `Components`: One array of draws per component: the base (None without a trend), the seasonality
  (None without a season), and each effect's contribution.
- `Control`: A control of the panel, its coefficient held to `lower <= 0 <= upper`.
- `decompose_dlm`: Draw `fit`'s states given all its data, `draws` times, and split each draw's mean
  into the base, the seasonality and each effect's contribution.
- `DegenerateFitError`: The data leave the margin regression degenerate; the message says how.
- `diagnose_dlm`: Test `fit`'s one-step errors, watch them for drift, and check its decomposition.
- `DiscountGrid`: The discounts to try, per family of blocks: the trend's, the season's, and the
  effects', one shared by every effect or with `PerEffect` each effect's own.
- `DlmChoice`: Every spec of a grid filtered on one geo, the one-step likelihood's best, and the
  specs the likelihood ratio keeps, each with its fit.
- `DlmDecomposition`: A fit's components over its periods, `(draws, periods)` each, from the
  posterior of its states.
- `DlmDiagnostics`: A fit's checks.
- `DlmFit`: The model fitted to one geo's calendar.
- `DlmForecast`: The KPI over the periods after a fit's data, forecast 1 to `horizon` steps ahead.
- `DlmForecastDraws`: Draws of the KPI over the periods after a fit's data, `kpi` `(draws,
  horizon)`, and the share of the paths drawn whose signed coefficients kept their signs at every
  period forecast, `acceptance`: how hard the signs bind over the horizon, 1 for a fit without them.
- `DlmSpec`: The model's structure.
- `Edge`: An end of a grid that a kept spec reaches where the parameter goes on past it, so the set
  the likelihood ratio keeps may stop there short of the data's: `parameter` names it, such as
  `"trend discount"`, `"tv discount"` or `"tv half-saturation"`, `end` is the grid's value there, a
  half-saturation point's as a multiple of the channel's mean positive spend, and `best` says
  whether the likelihood's best spec sits there too.
- `Effect`: A channel or control of the panel, with a random-walk coefficient and its discount.
- `Finding`: A plausibility check the decomposition fails: on which effect, if one, and on which
  dates.
- `fit_bayes`: Draw the posterior of `spec` on one geo of `panel` over `window`, its first and last
  dates, by default the whole calendar, its likelihood holding `tests`, each channel's geo tests,
  `estimates`, each channel's lift estimates, and `series`, each channel's lift series, if given.
- `fit_bayes_geo`: Draw the posterior of `spec` on every geo of `panel` over `window`, its first and
  last dates, by default the whole calendar, its likelihood holding `estimates`, each channel's lift
  estimates, if given.
- `fit_dlm`: Filter and smooth one geo of `panel` through the model `spec` describes.
- `fit_geo_dlm`: Fit `spec` to every geo of `panel`, the geos' spread chosen by the marginal
  likelihood.
- `fit_margins`: Each channel's margin over the weekly history: `sales` `(weeks,)`, `spend` `(weeks,
  channels)`, the `promotion` flag and the `price` `(weeks,)`, either None where the history holds
  none, and the adstock's `kernel` length in weeks; under the mixture of error models `noise`,
  `MIXTURE` by default, and with None as the research code fitted it, whose margins the sales' level
  moves.
- `fit_nested`: Draw both stages' posteriors on one geo of `panel` over `window`, its first and last
  dates, by default the whole calendar, as `nested_model` reads `estimates`, each channel's lift
  estimates on the KPI, and `mediator_estimates`, each channel's on the mediator: apart, each stage
  from a seed spawned from `seed`, or, where a lift on the KPI is of a channel of stage one, as one
  model from a third.
- `fit_pooled_geos`: The model over geos (the module) fitted to `panel` with each channel's lift
  `estimates`, its kernel `length` weeks long.
- `fit_ridge`: Fit `spec` to one geo of `panel` over `window`, its first and last dates, by default
  the whole calendar, calibrated by `tests`, each channel's geo tests, if given.
- `forecast_dlm`: Forecast the `horizon` periods after `fit`'s data, every effect's column taken
  from `plan`.
- `forecast_draws`: `draws` draws of the KPI over the `horizon` periods after `fit`'s data, the
  effects' columns taken from `plan` as `forecast_dlm` takes them.
- `FourierSeason`: Harmonics 1 to `harmonics` of a season `period` periods long (52.18 weeks to a
  year is allowed), counted from the window's first period: `season_cos_h` and `season_sin_h` each.
- `geo_effects`: Each geo's coefficients at the last period, the national one plus its deviation,
  with the spread's uncertainty: their mixture over the spread's draws, read as one Student t with
  the mixture's variance and the last step's degrees of freedom.
- `GeoDlmFit`: The model fitted to every geo of a panel.
- `GeoDlmSpec`: The model's structure over geos.
- `GeoEffects`: Each geo's coefficient on each effect at the last period, `(geo, effect)`: its mean
  and the central interval holding `level` of it, mixed over the spread's draws.
- `Lifted`: A fit's posterior given its KPI and its tests.
- `LiftReading`: The tests a fit reads, each an observation of one coordinate of the state path:
  test `j` reads coordinate `coordinates[j]` at period `t` with weight `weights[j, t]`, zero outside
  its readout, and its lift `lift[j]` has standard error `se[j]`, both at the market's scale.
- `LinearTrend`: A line through the window's periods, bent at `knots` points spaced evenly inside
  it: the period's place `t` in the window, `trend`, and a hinge `max(0, t - knot)` for each knot,
  `trend_bend_1` on.
- `MarginFit`: The margins' posterior, the mixture over the retention combinations.
- `Media`: A channel of the panel, its spend carried over by `adstock` and saturated by `hill`.
- `Mixture`: A channel's posterior over its grid, the other channels at their points: `points`, the
  kept points as indices into `POINTS`; `weights`, theirs, summing to 1; and `coefficients`, the
  posterior mean of the channel's coefficient in each geo under each point, `(points, geos)`.
- `NestedFit`: Both stages' posteriors over one geo's window: `stage_one` the mediator's, its
  returns in the mediator's units per currency unit, its spec's priors those it was drawn under, and
  `stage_two` the KPI's, the mediator's coefficient among its controls'.
- `NestedSpec`: The two stages and the control of the panel that mediates between them.
- `NoiseMixture`: The regression's settings under a mixture of error models (the module): the weeks
  between the trend's interior knots, `knot_every`; the AR(1) coefficients `persistence` and the
  AR(1)'s shares of the errors' variance `share`, each pair of them an error model, with white
  errors once more where `white`; and the promoted weeks' variance ratios `promoted`, each crossed
  with every one of those.
- `PerEffect`: Discounts each effect takes on its own: a grid tries every combination of `values`
  over the effects, the last effect's varying fastest, and counts one parameter an effect.
- `PooledGeoFit`: The fit of a panel over geos (the module).
- `RidgeFit`: The model fitted to one geo over a window of the panel's calendar.
- `RidgeSpec`: The model: its channels, its controls, its trend and season if any, and the ridge
  `penalty > 0`.
- `search_media`: Choose each transformed effect's transform on `grid` and the discounts on
  `discounts` by the one-step log-likelihood on one geo, and fit every spec filtered that a
  likelihood-ratio test at `level` does not reject, or with `fitted` at most that many of them.
- `Season`: Harmonics 1 to `harmonics` of a `period` counted in the panel's periods (52.18 weeks to
  a year is allowed), with one discount.
- `Statistic`: A test's statistic and its p-value under the model.
- `TransformGrid`: The transforms a search tries for each channel: `retentions`, `half_saturations`
  as multiples of the channel's mean positive spend on the geo, the Bayesian model's unit for them,
  and `shapes`.
- `transforms_from_bayes`: Each channel's transform at its parameters' posterior medians, by
  channel, as `BayesFit.channels` reads them: its retention, its half-saturation point, its curve, a
  Hill's slope at its median, and the fit's kernel length.
- `Trend`: A level (`order=1`), or a level and a slope (`order=2`), with one discount.
- `vague_geo_prior`: The engine's default prior over geos: weakly informative, on each geo's own
  scale.
- `vague_prior`: The engine's default prior: weakly informative, on the series' own scale.

### `mixdyn.models.demand`

- `DemandProxy`: The panel's control `column`, a demand proxy such as query volume, in the one
  `role` this model gives it; `searched` names the channels whose spend follows the demand it
  measures.
- `Reading`: What a channel's return is under a demand proxy's role; see the module.
- `Role`: What a demand proxy is to the model: a control, `"confounder"`, or left out, `"mediator"`.

### `mixdyn.transforms`

- `Adstock`
- `carry_over`: The adstock of a `(T,)` spend series, its kernel as long as the series.
- `channel`: The channel `coefficient * saturate(carry_over(spend))` in `chc.response`'s terms, its
  kernel `length` periods long.
- `Geometric`: Each period keeps `theta` of the carryover it received, `0 <= theta < 1`.
- `half_saturation`: `K`, `gamma` times the most of a window's adstocked values.
- `Hill`: A Hill curve of slope `alpha > 0`, its half-saturation point `gamma` times the most of the
  window's adstocked values, `0 < gamma <= 1`.
- `HillShape`: `a^n / (a^n + K^n)`: half its ceiling at `K`, S-shaped where its `slope` `n` passes
  1.
- `robyn_weibull`: Robyn's CDF Weibull adstock of `shape >= 0` and `scale` in `[0, 1]`, a share of a
  series of `length` periods.
- `saturate`: The Hill curve of slope `hill.alpha` and half-saturation point `half` at adstocked
  values.
- `Shape`
- `TanhShape`: `tanh(atanh(1/2) a / K)`: concave from no spend, half its ceiling at `K`.
- `Transform`: A channel's spend carried over by a normalised geometric kernel of `retention` and
  `length` periods, then saturated by a curve of `shape` that is half its ceiling where the adstock
  is `half_saturation`, in the spend's currency.
- `WeibullCdf`: The carryover that survives each lag, a Weibull survival of `shape` and `scale`
  periods.
