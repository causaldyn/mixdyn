# 0060. The health card

Status: accepted, 2026-10-31; amended 2026-11-06

## Context

A Bayesian fit reported its convergence, each parameter's R-hat and effective sample sizes, its
divergences and its E-BFMI, and nothing on whether the model fits the data or what its priors
did. Meridian 2.x (Google) ships a model reviewer whose health checks a buyer compares with any
other tool's: the largest R-hat, a posterior predictive p-value of the total outcome, the
probability of a negative baseline, and how far each channel's return moved from its prior. Two
of those catch failures convergence does not: a model that cannot reproduce the KPI's total, and a
baseline the channels have pushed below 0.

## Decision

- **`health(fit, panel, geo, *, seed, prior_draws=4000)`** reads a one-geo fit
  (`mixdyn.models.bayes_mmm`) into a `HealthCard`, each cell a number:
  - `max_rhat`: the largest R-hat over every parameter, a constant's `nan` left out.
  - `ppp`: the two-sided posterior predictive p-value of the KPI's total over the fitted periods,
    `min(1, 2 min(P(T >= t), P(T <= t)))`, `t` the observed total and `T` a replicate's. One
    replicate at each draw of the posterior, the KPI drawn about the draw's mean by the model's
    noise at that draw: normal, Student-t, or log-normal about the mean (ADR 0017), whose sum has
    no closed form. `ppp_se`, `2 sqrt(q (1 - q) / S)`, `q` the smaller tail's share of `S`
    replicates, is its Monte Carlo error from the draw count; it reads 0 where no replicate is past
    the total, and the p-value is then below `2 / S`.
  - `negative_baseline`: the share of draws whose baseline, the intercept, calendar and controls
    without the channels (the level less media, ADR 0036), is below 0 at some period of the
    window; `negative_baseline_periods` that share period by period.
  - per channel, of its return as ADR 0016 defines it, the window's spend's revenue with its
    carryover counted, at the posterior's draws and at `prior_draws` draws of the prior:
    `shift`, the posterior mean less the prior's over the prior's standard deviation; `width`, the
    posterior's standard deviation over the prior's; and `consistency`, the prior's quantile of the
    posterior median.
- Each draw's mean is the fitted mean's formula of ADR 0028 at that draw, each channel built as
  `BayesFit` builds it; averaged over the draws it is the fit's own posterior mean, a test to
  `1e-12`. Each draw's return is `mixdyn.decisions.returns`' of `BayesFit.channel_draws`' channel,
  a test at seven draws to `1e-12`.
- **The prior of a return** is drawn from the priors the model states: the return as stated, the
  decay `Beta(1, 3)`, the half-saturation point `LogNormal(0, 1)` in the spend's unit and a Hill's
  slope `LogNormal(0, 0.5)`, the coefficient set by the return as the model sets it. A return
  stated over the window is not ADR 0016's, whose prior is then the stated one carried past the
  window's end through the drawn decay and curve; a carried return's is the stated draw itself, a
  test exactly.
- **`prior_predictive(panel, geo, spec, *, draws, seed, window=None)`** draws the same prior and
  returns `PriorPredictive`: the media share, the channels' contribution over the window over the
  KPI's, and each channel's return, with their `quantiles`. It is the stated prior, not its
  truncation to the support under log-normal noise (ADR 0017).
- **`HealthCard.flags(...)`** reads each cell against a threshold the caller passes, and no cell
  whose threshold is not passed: `max_rhat`, `min_ppp`, `max_negative_baseline`, `max_width` and a
  consistency `tail`. Meridian's reviewer reads a largest R-hat below 1.2, a p-value of 0.05 or
  more and a probability of a negative baseline below 0.2; the docstring cites them as the
  example, `flags(max_rhat=1.2, min_ppp=0.05, max_negative_baseline=0.2)`. None is a default.
- The cells' arithmetic is public on arrays, for any engine's draws: `posterior_predictive_p`,
  `negative_baseline` and `prior_shift`.

## Consequences

- *Three worlds* drawn from the one-geo model, two channels over 104 weeks, the KPI log-normal
  about its mean at a log scale of 0.05, each fitted by four chains of 250 draws:

  | world | largest R-hat | ppp | P(negative baseline) | flags at Meridian's thresholds |
  |---|---|---|---|---|
  | well-specified, level 855 | 1.0084 | 0.982 ± 0.032 | 0.000 | all pass |
  | six weeks of outage at a thousandth of the KPI, no column for it | 1.0082 | 0.004 ± 0.003 | 0.000 | ppp fails |
  | level −300, the channels' spend tenfold | 1.0181 | 0.944 ± 0.032 | 1.000 | negative baseline fails |

  On two more seeds of each world the well-specified one passes every cell, the outage world's
  p-value is at most 0.008 and the negative level's probability is 1.000.
- *The p-value on the total is weak by construction where the noise is additive*: an intercept
  fitted by a normal likelihood makes the fitted mean's total the observed one, so the replicates
  centre on it whatever the model misses. It fails where the noise is read on the log, as above:
  the outage weeks widen the log's scale, and the mean the model reads about it overshoots the
  total. Four weeks of outage at a hundredth of the KPI read 0.082, six 0.020 to 0.022, eight
  0.020 to 0.044: below a thousandth it fails at every seed tried.
- *A confounder can move another cell instead.* Outage weeks in which the channels' spend also
  stopped read a p-value of 0.996: the posterior gave search a return of 30 and the level 8, and
  search's consistency, 0.99975, is what flags it.
- *The default priors' media share*: on the well-specified world, whose spend is 8.3 % of its
  revenue, the share's 90 % prior interval is 0.039 to 0.371, inside (0, 1); each return's prior
  median is near `exp(0.2)`.
- *The pooling guard*: a scale's prior swapped from a half-normal to a log-normal, the noise
  scale of the one-geo model and of each stage of the nested model, each channel's spread in the
  model over geos, moves the density by more than a constant and leaves the coordinates and the
  axes of every parameter drawn on it, shared, per geo or per channel, as they were.
- *Left*: the model over geos and the nested model have no card: their arithmetic is the
  functions' above, their means the models' own. The DLM has its own diagnostics (`mixdyn.models.dlm_diagnostics`).
- *What a validation of the flags will need.* The scorecard is to measure each flag's AUROC at
  telling a fit whose plan or returns are wrong from one whose are right, over worlds where the
  truth is known. It needs each cell as a number, not a verdict, so that the curve sweeps the
  threshold; the p-value's Monte Carlo error, to tell a threshold from the noise in reading it;
  the same draws and seeds as the fit's record, so that a cell can be read again; and the
  per-channel cells keyed by channel, to be scored against each channel's error.

## Alternatives considered

- **Meridian's thresholds as defaults.** Rejected: they are one tool's, not measured on this
  engine's fits; the AUROC validation is to set them, and until then a caller states them.
- **The p-value of the total's posterior mean**, without the noise. Rejected: the observed total
  carries the noise, which the mean's spread leaves out, so a well-specified model would read
  below a threshold more often than the threshold says. Not measured here.
- **The shift of the parameter the prior is stated on.** Rejected for the card: ADR 0016's return
  is the one every report reads, and for a carried prior it is that parameter.

## Amended 2026-11-06: the flags validated, the plan's range, the pooling guard

- **`out_of_range(history, plan)`** (`mixdyn.inference.diagnostics`) reads how far a plan's
  spend lies outside the range the history logged, the same way on both sides: past a channel's
  most spend a period, the spend less that most; short of its least, that least less the spend.
  Summed over the plan's periods and channels and divided by its whole spend, it is 0 inside
  every range and passes 1 where a plan spends further below the least than it spends; it
  returns each channel's part beside it. Counting all of a short spend instead would read a plan
  just below the least as far out as one at 0, and one side differently from the other. It is a
  function on arrays, as the card's arithmetic is, so no output of a plan moves.
- **The pooling guard** also runs over the model over geos' own prior options, `noise_prior` and
  `base_prior`, each alone and both: the coordinates by parameter and index, a base sampled as it
  is rather than on its log, and every drawn parameter's axes stay the default's.
- **The scorecard reads the flags as classifiers** (`scorecard.flags`) in a run's report asked
  with a `tau` (`report --flags TAU`), at each rung over the look every environment holds whole;
  a report asked without one prints as it did, to the byte. Every flag a record holds that a stated
  direction names (a status reads PASS, REVIEW, FAIL in that order) is scored by its AUROC against
  two outcomes of each world: a channel's ROI error above `tau`, the mean of the arm's draws
  against the world's, a channel's flag read against its own channel and a fit's against some
  channel; and the plan's regret above the status quo's. The interval is the percentile bootstrap
  over worlds, 2 000 resamples from one seed, a world's channels drawn together. A flag below 0.5
  is dropped from the report with the reason; one whose interval lies above 0.5 is said to tell
  the outcomes apart. The tools' flags are read the same way: Robyn's NRMSE, DECOMP.RSSD and R²,
  Meridian's R-hat, its reviewer's statuses and the numbers its checks keep (the posterior
  predictive p-value, the probability of a negative baseline, R²), PyMC-Marketing's R-hat and
  divergences, Orbit's ELBO drift. `tau` is the reader's: 0.5 (`scorecard.flags.TAU`) is a
  placeholder, and a scored run reads the value its pre-registration freezes; the run's
  declaration does not hold it.
- **R² against regret**: each record's R² in sample is read against its plan's regret per euro by
  Spearman's rank correlation, with the same bootstrap's interval.
- **What the records now hold**: the engine's static arm records each family's card, `ppp`,
  `negative baseline`, each channel's `prior shift`, `prior width` and `prior consistency`, its
  replicates and prior drawn from a child of the export's seed apart from the fits' and the
  forecast's, and its R²; the ridge arm its R²; and every plan of one weekly spend a channel its
  `out of range` share and each channel's. The plans do not move. The model over geos has no
  card yet, and neither it nor the DLM records an R²; a geo export's cells record no range share,
  each cell's range being its own.
- *Read on a pilot of family 2* (ten worlds of each of three environments, records of 2026-10-30,
  before the card was recorded), at `tau` 0.5: at rung 0 the CHC arm's flag that a channel's scale
  sits at an end of its grid read 0.898 [0.856, 0.934] against a wrong return, and the ridge arm's
  coefficient on its bound 0.713 [0.676, 0.750]; Meridian's posterior predictive p-value read 0.230
  [0.075, 0.410] against regret, below 0.5 and dropped, and its R² against regret a rank
  correlation of +0.079 [-0.304, +0.444]. Thirty worlds are a pilot: most intervals hold 0.5.
- *Not done*: the thresholds the card is read at are still the caller's; the scored runs that will
  set them have not run.
