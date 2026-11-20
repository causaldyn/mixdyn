# 0079. A blocked holdout, a tempered refresh, and the refresh cell

Status: proposed, 2026-11-07; adds `holdout_bayes` and `refresh_bayes` to the one-geo model
(`mixdyn.models.bayes_mmm`, at the module's path, as `fit_bayes` is) and the scorecard's refresh
cell (`scorecard.refresh`); amends ADR 0040; no default and no existing output changes
Amended 2026-11-07: the DLM arm's pilot on every swing world (below)
Amended 2026-11-16: the rivals' arms on each cut, and a cell run from a plan (below)

## Context

A static model is fitted once to a window, and read for returns and a plan. Two questions follow
that the engine had no function for:

- **How well does it forecast periods it did not read?** ADR 0040's forecasts are a bootstrap of
  a run's items, and hold out no periods. A holdout that draws its periods from the whole window
  leaves each one between fitted neighbours. Where the KPI's noise is autocorrelated, a term that
  bends with time fits the noise the held period shares with them, and the score reads a forecast
  from inside the window, one no planner gets.
- **What happens when weeks arrive?** A planner fits again on a longer history. The returns move,
  and with them the plan. How far they move between refits on a world that holds still is the
  model's own instability, a cost to the planner that the scorecard did not measure. The engine
  had no way to carry the previous posterior into the next fit either.

## Decision

### A holdout, blocked by default (`holdout_bayes`)

- `holdout_bayes(panel, geo, spec, *, periods, seed, kind="blocked", ...)` masks the KPI at
  `periods` of the window's observed periods, as periods the panel did not observe, fits `spec`
  by `fit_bayes`, and returns a `BayesHoldout`: the held dates, the KPI there, the posterior
  predictive's draws there, one at each posterior draw, and their CRPS averaged over the periods,
  over the KPI's mean over the window's observed periods. The window, the trend's bends, the
  season and the carryover are the fit's own.
- The predictive is the model's mean at each draw, the panel's spend and controls at the held
  periods, with the KPI drawn about it by the model's noise, as `forecast_bayes` draws it (one
  function, `_noised`, now does it for both). The CRPS is the sorted-draws estimator of
  `E|X - y| - E|X - X'| / 2`. The fit's, the periods' and the noise's seeds are children of
  `seed`.
- **`kind="blocked"`, the default, holds out the window's last periods.** `"interleaved"` draws
  them from all of them; it stays, to show the leak.

### A refresh by a power prior (`refresh_bayes`)

- `refresh_bayes(panel, geo, previous, *, seed, weight=WEIGHT, level=0.8, spec=None, ...)` fits
  the model again to the whole panel, the previous window included, with each channel's return
  prior tempered by the previous posterior, and returns a `BayesRefresh`: the new fit, the weight,
  and each channel's `ReturnDrift`.
- **The tempering is a power prior** (Ibrahim and Chen, 2000) on each channel's log return. The
  previous posterior's draws of the log return give a normal `N(m, s^2)`. Over the previous prior
  `N(mu_p, sigma_p^2)` that is the previous periods' evidence, a normal of precision
  `1/s^2 - 1/sigma_p^2`; none where that is not above 0, as on a channel the data did not narrow.
  The base prior `N(mu_0, sigma_0^2)` times the evidence to the power `a` is normal: precision
  `P = 1/sigma_0^2 + a (1/s^2 - 1/sigma_p^2)`, mean `(mu_0/sigma_0^2 + a (m/s^2 - mu_p/sigma_p^2))
  / P` (`tempered_spec`; Maxima, `validation/refresh_prior.mac`, steps 1 and 2).
- **Why a power, and why small.** The refit reads every period the previous fit read. Its
  likelihood already holds their evidence, so the tempered prior counts it `1 + a` times. At
  `a = 0` the refresh is the full refit, draw for draw. Above 0 it damps the move a few new weeks
  make, and pays for it by counting old weeks more. Under normal evidence the refreshed mean lies
  at most `a / (1 + a)` of the way from the full refit's towards the previous periods' own
  reading, and its SD is at least `1 / sqrt(1 + a)` of the full refit's (steps 3 and 4). **The
  default, `WEIGHT = 0.25`**, bounds these at a fifth of the way and 0.894 (step 5).
- **Only the returns are tempered.** The intercept, trend, season and controls are the window's
  own and are refitted from their base priors: a trend of the previous window is not one of the
  new window's.
- **The base prior is `spec`, `previous.spec` by default.** A chain of refreshes passes the first
  spec, so the evidence of each refresh's own periods is tempered into the base once, not
  compounded. Only a log-normal return prior (`RoiPrior`) is tempered; a `HalfNormalRoi`, whose
  log is not normal, is refused, in `spec` or in the previous fit's.
- **The drift** is each channel's return on the previous window's spend, as its prior counts it
  (`returns_on`): `before`, the previous posterior's mean, with its central `level` interval, and
  `after`, the refreshed mean of the same return. `leaves` is whether `after` lies outside the
  interval.

### The refresh cell (`scorecard.refresh`, the bench's `causaldyn_bench.scorecard.refresh`)

- **The swing.** A world's export is cut to its first `T - 4 (6 - i)` weeks for `i = 0, ..., 6`:
  a first fit on all but 24 weeks and six refits, each on four more. Every cut plans the world's
  one quarter, so every plan is scored on one truth. A cut's controls over the quarter are the 13
  weeks after it; its lift tests are those whose readout ends inside it. Each fit's returns are
  read from its mapped response on one window, the year ending at the first cut's last week.
  Read: for each channel, the largest move of the mean return between consecutive refits and the
  refits whose mean leaves the previous fit's central 80 %; for the plans, each refit's regret per
  euro on the truth less the previous fit's.
- **The rolling origin.** A world whose effect holds still is cut at the same seven lengths, each
  cut a world with its own quarter and truth. On the short-history family an arm reads the last
  `L` weeks before each origin, so its window rolls. Read: the swing's columns on the weeks every
  window holds, and each origin's CRPS of its forecast of its quarter at the status quo.
- `python -m scorecard.refresh run` fits one arm on every cut of one world in one process and
  keeps each cut's records; a stopped run resumes at the cut it stopped at. `report` reads every
  complete world. The cell runs on pilot worlds and reads no declaration: its numbers state what
  moved, not a claim.

## Consequences

- **The leak, measured.** Four worlds of 104 weeks, two channels, 13 weeks held out, a trend of 12
  knots, 2 chains of 200 draws: with AR(1) noise at 0.95 the interleaved holdout's CRPS was 0.07
  to 0.28 of the blocked one's, and 0.19 to 0.42 of its own on the same worlds with independent
  noise of the same SD (`tests/test_bayes_refresh.py` holds world 1 below 0.5 and 0.6). The
  blocked holdout's CRPS on independent noise was 2.6 to 4.6 times the interleaved one's too: a
  trend of 12 knots also forecasts badly past its window, which the blocked holdout reads and the
  interleaved one hides. With 3 knots, the scorecard's arm's, the blocked CRPS was 1.10 and 1.34
  times the interleaved on two of the worlds: the leak needs a term that bends enough to follow the
  noise.
- **A refresh is not the full refit at the default, by design.** On Track M v2's drawn and
  reference worlds 900, each continued by its quarter, four chains of 1000 draws
  (`evidence/bench/refresh_continuation.py`, its record
  `evidence/bench/results/refresh_continuation.json`, mixdyn `33ed7a5`, the bench `ea24f2d`): each
  channel's mean log return lay 0.11 to 0.33 of the full refit's SD from the full refit's, against
  0.01 to 0.10 between two seeds of the full refit. **The stated tolerance is half an SD.** No
  refreshed mean left the previous 80 % interval.
- **The normal bounds are the closed form's, not the posterior's.** Paid shopping's SD ratio was
  0.68 and 0.73, below the 0.894 the normal bound allows, and meta's mean moved 0.14 to 0.22 SD
  though its evidence was none and its prior was not tempered. The returns are read jointly: a
  channel whose prior moves moves the others that share its periods' sales. The likely reason for
  paid shopping's narrower SD is a posterior far from normal on the log return, whose tail towards
  0 the shifted prior trims; this was not measured.
- **The cell's pilot** (`evidence/bench/results/refresh_pilot.json`; the fits ran at mixdyn
  `33ed7a5` and were read at `0ab5849`, which differ in documents alone; the bench `ea24f2d`). The
  swing on Track M v2's drawn and reference worlds 900 and the drift family's static and walk
  worlds, the rolling origin on the short family's first worlds at 52 and 104 weeks; the ridge,
  CHC and Bayesian arms, each at its full sampler:
  - **The Bayesian arm's stacked posterior, the engine's default, never left its 80 %**: on all six
    worlds, in 108 pairs of consecutive fits a channel, no refit's mean return lay outside the fit
    before's central 80 %. Its largest move of a channel's mean was 0.12 to 1.8 a euro on the swing
    and 0.25 to 3.2 on the rolling origin. On the four swing worlds its plan's regret did not move
    between refits: it stayed at the first fit's, 0 to 1.20 a euro. From a rolling origin it moved
    by at most 0.074 a euro, and the quarter's CRPS averaged 0.17 at 52 weeks and 0.14 at 104.
  - **The ridge arm left its 80 % on 5 or 6 of 6 refits** of paid shopping and meta on every world:
    its draws are narrower than its moves. It read television's return as 0 on every swing fit.
  - **The CHC arm's returns moved most**: its largest move of a channel's mean was 3.1 to 70 a euro
    a world, where the swing worlds' true returns on the window are 2.2 to 5.2; its plan's regret
    moved by up to 0.46 a euro.
  - The DLM arm ran on the drawn world's swing alone, and its sixth fit was stopped by the 5 GB
    memory cap while it drew its unmixed chains again at 2000 draws. On its first five fits its
    mixture's means moved by up to 8.1 a euro and never left the fit before's 80 %.
- `BayesHoldout`, `ReturnDrift` and `BayesRefresh` are saved by `mixdyn.api.serialize`; adding
  them moves no format version.

## Alternatives considered

- **An interleaved holdout by default.** It scores more periods of the window and needs no
  forecast past it, and it reads the leak. Rejected.
- **A holdout with a buffer around each held period** (an h-block). It mends the leak where the
  noise's memory is short and known. The blocked holdout needs neither, and scores the forecast a
  planner makes.
- **The previous posterior as the prior of a fit to the new weeks alone.** The static model's
  intercept, trend and season are the window's: a fit to four weeks reads none of them, and the
  previous posterior's marginals drop the correlations the returns hold with them. Rejected.
- **Weight 1.** The previous periods' evidence then counts twice in full, and the refreshed mean
  can lie halfway towards the previous reading. Too strong to call a refit.
- **Tempering every parameter**, by a normal over the whole posterior. The posterior is not normal
  in the parameters the sampler moves, and the returns are what a planner reads. Not taken.
- **Reading each refit's returns on its own window.** The windows differ, so a return's move would
  mix the spend it is read on with the reading. Rejected: the swing reads every fit on one window.

## Amended 2026-11-07: the DLM arm's pilot

The DLM arm's swing ran again on the four swing worlds above, every fit in, under a memory cap of
10 GB (`evidence/bench/results/refresh_pilot_dlm.json`; mixdyn `d5a879f`, the bench `95a61f9`,
CHC 1.2.0, which the record does not name).

- **The memory.** 28 fits, none failed. The process's peak resident memory, as the records' cost
  states it, rose with the fits to 4.8 to 5.6 GiB a world. On the drawn world it passed 5 GiB at
  the sixth fit, 4.97 to 5.55 GiB, the fit whose unmixed chains were drawn again at 2000 a chain and
  which the 5 GB cap stopped before. The drawn world's first five fits are the earlier run's
  records, every field but their cost and versions.
- **The mixture's returns moved further than the stacked posterior's, and its 80 % held them.** Its
  largest move of a channel's mean was 0.8 to 19 a euro, against the stacked posterior's 0.12 to
  1.8; the refit's mean left the fit before's 80 % in 4 of 72 pairs of consecutive fits a channel,
  all on the static world of the drift family. Its means lay at 0.9 to 43 a euro, where the swing
  worlds' true returns on the window are 2.2 to 5.2.

  | world | the mixture's largest move a channel: shopping, meta, television | leaves | its plan's regret, first to last fit | largest regret move: mixture, plug-in, CVaR 0.1 |
  |---|---|---|---|---|
  | Track M v2, drawn 900 | 0.78, 8.1, 6.5 | 0 of 18 | 0.115 to 0.002 | 0.086, 0.116, 0.027 |
  | Track M v2, reference 900 | 3.4, 18.9, 1.7 | 0 of 18 | 0.105 to 0.605 | 0.476, 0.546, 0.054 |
  | drift, static 220500 | 2.6, 15.9, 6.4 | 4 of 18 | 0.010 to 0.010 | 0.033, 0.035, 0.009 |
  | drift, walk 220000 | 2.2, 9.2, 0.79 | 0 of 18 | 0.593 to 0.797 | 0.162, 0.121, 0.016 |

- **Its plan moved with its refits.** Where the stacked posterior's regret stayed at its first
  fit's on every swing world, the mixture's moved by up to 0.48 a euro between consecutive fits,
  and its plug-in's by up to 0.55; the CVaR plan at 0.1 moved by at most 0.054.

## Amended 2026-11-16: the rivals' arms on each cut, and a cell run from a plan

The cell refitted this engine's arms alone. A rival's stability between refits is what its vendor
sells, so the cell now runs the tools' arms too, and a whole cell runs from one plan.

- **A tool's arm on each cut** (`fit_tool`). Each cut is exported as the bench archives a world,
  into a directory of its own, and the tool's script runs on it as `scorecard.run` runs a tool's
  world: from the bench's tree, in a scope capped at the arm's memory with no swap, stopped at its
  timeout, tried twice. A cut whose process left no record is kept as a failure, which plays the
  status quo, as a world's is. The records are read by the same `refresh.swing` and
  `refresh.rolling`.
- **`run --cut I`** fits one cut alone.
- **`cell --plan PLAN.json`** runs every job of a plan, each job one world's arm at one rung, each
  of its cuts a `run --cut` in a process of its own, admitted by the memory gate that
  `scorecard.run` uses. An engine arm's cut is tried twice, and a cut that left no record is kept as
  a failure of each variant. *Why a process a cut:* in one process for a world's seven fits the
  DLM arm's peak resident memory rose with the fits to 4.8 to 5.6 GiB (above); a process a cut
  holds one fit, so an arm's cap is its one fit's, as a run declares it, and a process that dies
  costs one cut.

No default and no existing output changes: `run` without `--cut` fits every cut in one process,
as before.
