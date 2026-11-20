# 0014. The nested model: a mediator between two Bayesian stages

Status: accepted, 2026-10-02; amended 2026-10-05 by ADR 0033: under a lift on the KPI of a
channel of stage one, which reads both stages, the stages are drawn as one model; amended
2026-11-06: on worlds where nothing television builds lasts in sales, its long-term share covers 0
on 91 of 100 where sales do not read the mediator, the criterion of 90 % met, and on none of 10
where television does not move it, and a dual adstock covers it on none, which is why mixdyn ships
none; amended 2026-11-07: stage one reads the default return prior at the mediator's scale, its
median times the mediator's mean over the KPI's, and a channel that may not move the mediator can
take a half-normal one (ADR 0076); amended 2026-11-07, later: a mediator that drifts needs a trend
in stage one, without which a channel's carryover takes the drift

## Context

Upper-funnel media, television or video, move sales partly through series the panel observes:
branded searches, site visits. A model that reads such a series as a control blocks the path, so a
channel that acts only through it reads no return. One that leaves the series out lets the
channel's return carry the path, but loses what the series says about the KPI and how the channel
moves it. A nested model's first stage can go wrong in two quiet ways: regressing the mediator on
a function of its own columns, so its R² is 1, and dropping the carryover it was asked for. The
engine needs both stages as full models of carryover and saturation, a channel's return through the
mediator by the same identity as its direct return, and the planner's curves.

## Decision

- **Two stages, each ADR 0010's model of one geo** (`mixdyn.models.nested`). Stage one reads the
  mediator, one of the panel's controls, as its KPI; stage two reads the panel's KPI with the
  mediator a control held at or above 0, its coefficient `gamma` the KPI's rise per unit of the
  mediator.
- **`NestedSpec(mediator, stage_one, stage_two)` refuses** a stage one that reads the mediator, as a
  channel or a control, which is the first defect above, and a stage two that does not read it held
  at or above 0. `fit_nested` refuses a panel without the mediator among its controls.
- **Fitted apart, paired draw by draw.** The stages share no parameter and the mediator is observed,
  so the joint posterior is the product of the two. Each is drawn by `fit_bayes` from a seed spawned
  from the caller's, and a draw of one with the same draw of the other is a draw of both.
- **A channel's return on the KPI is its direct return plus `gamma` times its return on the
  mediator**, at each draw. By stage one's return identity a channel contributes its return times
  its spend to the mediator over the window; the mediator enters the KPI linearly, so the
  contribution through it is `gamma` times that. Both stages read the same window, and the periods
  the panel filled are left out of both.
- **The planner** reads a channel of stage two as that stage hands it, and a channel of stage one as
  stage one hands it, its coefficient times `gamma`'s median. A channel in both stages has a sum of
  two curves for its return, which `chc.response` does not read: `channels()` refuses it, and
  `roi()` adds its paths.
- **Stage one's default return prior at the mediator's scale** (amended 2026-11-07). Stage one's
  returns are in the mediator's units a currency unit. The engine's default, `RoiPrior()`, is
  Meridian's for a revenue KPI, a median of 1.2 of the KPI's units a currency unit; read as stated
  in stage one, it says 1.2 searches or queries a euro whatever the mediator counts. A stage-one
  channel left at `RoiPrior()` now reads it at the mediator's scale: its location moved by
  `log(M / Y)`, `M` and `Y` the mediator's and the KPI's means over the window's observed periods,
  so its median is 1.2 times `M / Y`. `HalfNormalRoi()` (ADR 0076) is read with its scale times
  `M / Y`. A prior stated otherwise is read as stated, and the fit's `stage_one.spec` holds the
  prior it was drawn under.
  - *Why a scale proportional to `M`.* Counting the mediator `c` times over multiplies `M` and
    every return on it by `c`, and moves the log return by `log c`; the prior's location moves
    with it, so stage one's posterior on its own scale does not move with the mediator's unit
    (Maxima, `validation/half_normal_roi.mac`, step 5; `tests/test_nested.py`, the model's log
    density at units of `1e-3` and `1e3`, to `1e-12`).
  - *Why `M / Y`.* It states for stage one what the default states for stage two: the return as a
    share of the stage's own mean, a currency unit's worth, has the same prior in both. A channel's
    return through the mediator, `gamma r_1`, then has the default's prior times `gamma M / Y`,
    the mediator's part of the KPI's mean: a path through a mediator that carries a tenth of the
    KPI is presumed a tenth of a direct return.

## Consequences

- *On the test's world* (`tests/test_nested.py`: 104 weeks; television moves searches by 0.5 a euro,
  carried over at 0.7; a search moves sales by 2; search ads move sales by 1.5 a euro, directly,
  and price by −20 a euro; two chains of 200 draws from Pathfinder's starts, the default): stage
  one explains 0.88 of the searches' variance, its carryover's median is 0.60 against a prior mean
  of 0.25, and `gamma`'s 2.10, 90 % interval [1.93, 2.25]. Television's return on sales is 0.90,
  its 90 % interval [0.59, 1.30] holding the world's 1. A one-stage fit with the searches as a
  control reads it at 0.21, [0.08, 0.50]: the data read none, and the log-normal prior keeps it
  off 0.
- *The validation* (`evidence/nested_validation.py`, its rule stated in its docstring before it ran;
  its record `evidence/results/nested_validation.json`): 100 worlds of the tests' design with drawn
  returns, television's direct return 0 in half of them, fitted from uniform starts, the default
  when it ran. The 90 % interval for television's return on sales holds the world's in 94, the
  central 95 % of a binomial of 100 trials at 0.9 being [84, 95]. Its parts are not calibrated:
  where the direct path is 0, the log-normal prior on its return has no mass there, so its interval
  never holds 0 and the total runs 0.22 high; `gamma`'s interval holds 80 of 100, the two trading
  off along a ridge. A one-stage model without the searches reads the total as precisely, without
  the bias. The decomposition, the model's reason to be, needs a prior under which a direct return
  may be 0: a central interval of a posterior on `(0, inf)` never holds it.
- `gamma` is read as causal, as every mediation reads it: whatever moves the mediator and the KPI
  together and is not in stage two's calendar or controls biases it, and with it every mediated
  return. The model cannot test that. A channel's direct path is confounded the same way by what
  stage two leaves out.
- The stages' calibration is ADR 0010's model's, stage one without controls and stage two with
  them; a total return is a function of their draws.
- A spec without controls, stage one's usual case, found that `fit_bayes` returned draws of a
  calendar and controls the model did not have, zero columns wide, which JAX's batched map could not
  reshape. A fit now holds them where the model has them, as it holds a Hill's slope.
- *Long-term null worlds* (amended 2026-11-06; `validation/long_term_null.py`, its record
  `validation/long_term_null.md`, and `tests/test_long_term_null.py`). A channel's long-term share
  is its return through the mediator over its total, `gamma r_1 / (gamma r_1 + r_2)`, at each
  draw. The worlds are the test world's design with nothing television builds lasting in sales:
  its path through the searches is cut where sales do not read them, or where television does not
  move them. The searches' and sales' noise are 15 and 20 a week, and each model is drawn by two
  chains of 300 warm-up iterations and 200 draws, with no divergent transition in any fit. The
  share's posterior is on `(0, 1)`, so its central interval never holds 0 exactly. It is read as
  covering 0 where its 90 % interval's lower end is below 0.02, a fiftieth of the channel's return,
  set on a pilot of other seeds before the run.
  - Where sales do not read the searches, the share covers 0 on 91 of 100 worlds: 0.91, its Monte
    Carlo standard error 0.029, its 95 % Clopper-Pearson interval [0.836, 0.958] (Maxima's beta
    quantiles agree). The criterion set for the model, at least 90 %, met where that interval holds
    0.9 or lies above it, is met; the interval does not rule out a rate below 0.9 either, down to
    0.836. The first 20 worlds gave 17, 0.85, which twenty worlds do not tell from 0.9: a rate of
    0.9 gives at most 17 of 20 with probability 0.32. So the run was extended to 100 on the same
    seeds and settings, the 20 kept. The nine that miss have lower ends of 0.021 to 0.041. On three
    of them, refitted, `gamma`'s 90 % interval's lower end is 0.054 to 0.071 and its median 0.22 to
    0.29: the data read a rise of sales with the searches that the world does not hold. On world
    12, which covers, it is [0.003, 0.118].
  - Where television does not move the searches, the share covers 0 on none of 10 worlds, its
    lower ends 0.040 to 0.110 and its medians 0.11 to 0.26. Stage one's log-normal return prior
    has no mass near 0, so television's return on the searches stays off 0 where the data read
    none, and `gamma`, which the data read near 2, carries it into the share: the limit the
    validation above found for the direct path, on the mediated one. A long-term share is read
    beside stage one's return interval and the health card's `width` against its prior (ADR 0060),
    not alone. With stage one's return prior half-normal (ADR 0076; amended 2026-11-07, the
    record's second table), the share covers 0 on 87 of 100 such worlds, [0.788, 0.929], and on
    18 of 20 where sales do not read the searches.
- *Stage one's default prior at the mediator's scale* (amended 2026-11-07; television's total
  return on sales, its 90 % interval read against the world's):
  - On the search-demand world of ADR 0074 (`tests/search_demand_world.py`, seeds 0 to 9, 156 weeks,
    its stage one television on the queries with the default left; `M / Y` 86.7 to 88.1; two chains
    of 500 warm-up iterations and 500 draws), the default read as stated, a median of 1.2 queries a
    euro against the world's 48 to 52, held the total on 1 of 10 worlds, its medians 0.53 to 1.07
    against 1.25 to 1.29: the path through the queries read near 0, its share 0.012 to 0.025 at the
    median. Read at the queries' scale, a median of 107, it holds it on 7 of 10, its medians 0.82 to
    3.24. Two worlds read it 2.2 and 2.6 times too high, their stage one's interval of television's
    queries a euro [159, 292] and [152, 367]: the queries' demand runs an AR(1) of coefficient 0.8
    that stage one's independent noise does not hold, a likely cause not tested here. One reads it
    too low, and one stage one diverged on 41 of 1000 draws. Ten worlds give the rate a Monte Carlo
    standard error of 0.14.
  - On ten worlds of `tests/test_nested.py`'s design (data seeds 20261002 to 20261011, fit seeds 5
    to 14, television's prior on the searches left at the default; `M / Y` 0.25; two chains of 300
    and 200), the median moved from 1.2 searches a euro to 0.30, the world's 0.5, nearer on the log
    scale (0.50 from 0.89). The total was held on 10 of 10 worlds and is held on 9 of 10, the miss
    [0.16, 0.98]; its medians moved from 0.71 to 1.39 to 0.44 to 1.06. Ten worlds do not tell 9
    from 10 at a nominal 90 %.
  - Not moved: the test world's own spec and the Stan reference's (`evidence/stan_nested.py`, its
    fixture `tests/fixtures/oracles/nested_stan.json`) state television's prior on the searches,
    `LogNormal(0, 1)`, so their models and the oracle are what they were; the long-term null worlds
    state the same. Moved: the search ads' stage-one prior in the tests' crossed spec, read only
    through the model's own density, and the nested world of the calibration harness
    (`validation/sbc.py`), by `log(M / Y)`, 0.43, its median from 1.2 to 1.9. The harness draws its
    truths from the model's own prior, so it stays a test of calibration, but its record,
    `validation/sbc/nested-small`, was drawn under the old default and is not re-run with this
    amendment.
  - A prior equal to `RoiPrior()` field by field is read as left at it, so that one prior cannot
    be stated in the mediator's own units; any other can.
- *Stage one's slow term* (amended 2026-11-07, later). The two worlds above that read television's
  total 2.2 and 2.6 times too high are worlds 4 and 5 of the search-demand world. The likely cause
  named above is tested, and it holds in this form: the demand's drift goes into television's
  carryover. Measured by `scripts/study_endogenous_spend.py`'s `demand` study, on the same ten
  worlds with two chains of 500 warm-up iterations and 500 draws:
  - Stage one, as the study states it, has a season and no trend. The queries' organic demand runs
    an AR(1) that drifts over the 156 weeks. On worlds 4 and 5 its least-squares slope over them is
    +0.15 and +0.19 on the log scale, the two largest rises of the ten (read off the worlds' own
    draws).
  - A channel's kernel runs from the panel's first week with nothing spent before it. So at a
    decay near 1, television's carryover rises as the weeks fill the kernel. At 0.99, its curve at
    a half-saturation of its mean positive spend has a correlation with the week of 0.93 to 0.99
    on the ten worlds. Without a trend, it is the one term that can follow the drift.
  - On worlds 4 and 5, stage one's decay ran to medians of 0.99 and 0.93, where its Beta(1, 3)
    prior puts 0.1 % of its mass above 0.9. Television's return on the queries read 220 and 256 a
    euro, against about 51. At the world's own curve, least squares reads 9 and 83 there: the
    independent noise does not by itself make the high reading.
  - **A trend of 3 bends in stage one**, as the scorecard's Bayesian arm states its model (the
    study's role `nested, trend`), gives the drift a term of its own. The decay's medians on
    worlds 4 and 5 are then 0.36 and 0.40, and the returns on the queries 40 and 74. Television's
    total is held on 9 of 10 worlds, its medians 0.96 of the total, 0.64 to 1.31 over the worlds,
    against 7 of 10 and 0.64 to 2.56 without the trend. World 8's 41 divergent transitions are
    gone; every fit's R-hat is at most 1.016. World 9 reads 0.64 of the total with the trend and
    without it.
  - **The half-normal stage one** (ADR 0076; the roles `nested, half-normal` and `nested,
    half-normal, trend`) had the same cause on worlds 4 and 8. Its decay's 90 % interval reached
    up to 1.00 and 0.97, and its chains did not agree: 124 and 23 divergent transitions, R-hat
    1.37 and 1.35. With the trend, no fit has a divergent transition, every R-hat is at most 1.022,
    and the total is held on 9 of 10 worlds, against 8 of 10 without it.
  - A stage one is the caller's spec, so nothing moves by default. The module's docstring says
    that a mediator that drifts needs a trend in stage one. The rows the study stated before are
    reproduced to the digit.

## Alternatives considered

- **Both stages sampled as one model.** The same posterior, since it factorises, for a larger state
  for the sampler to adapt to.
- **Stage two reading stage one's fitted mediator**, two-stage least squares' way: `gamma` from the
  channels' variation alone, the channels its instruments. It is robust to what confounds the
  mediator and the KPI only where no channel also acts directly, and it discards the mediator's own
  variation.
- **The mediator through a curve in stage two.** A channel's return would no longer be the sum of
  its paths; not built until a panel needs a saturating mediator.
- **A channel in both stages handed to the planner** as a sum of two curves, which needs a channel
  `chc.response` does not have.
- **A dual adstock** (amended 2026-11-06): the channel entered twice, once as spent and once
  carried over at a long retention, 0.9 here, the second column's share of the two read as the
  long-term effect. Cain (2025, Applied Marketing Analytics 11(1)) finds it reads spurious
  long-term effects. On the long-term null worlds where sales do not read the searches, its share
  covers 0 on none of 20, its lower ends 0.063 to 0.140: the long column shares the direct path's
  own carryover, and each column's log-normal return prior keeps it off 0. Not shipped: a
  long-term effect is read through a series that carries it, the mediator.
