# 0068. Price, promotion and organic levers, measured against their counterfactual level

Status: accepted, 2026-11-06; adds `MediaPanel.levers`, read by the ridge and the one-geo Bayesian
model, refused by the DLM and the models over geos, and changes no fit of a panel without them.
Amended 2026-11-07: the DLM reads a lever too (below)

## Context

A panel's controls enter every engine as a column times a coefficient, so each control's
contribution is measured against 0. For a price that is no counterfactual anyone plans: a price
of 0 is not a policy, and "what did the price do" means against a price that was charged, its
highest or its regular one. A promotion's depth is read against none run, or against the least
run. Organic activity, newsletters sent or posts published, has no spend: it belongs in the
decomposition, and a planner that allocated a budget to it would be allocating to a column it
cannot buy. Meridian treats these as non-media treatments with a baseline of the observed minimum,
maximum or a given value, and organic media as reported but not optimised.

A promotion moved with the media also confounds them: a model without it credits the media with
its lift, and one with it can take a share of theirs.

## Decision

- **The panel declares levers.** `MediaPanel.levers` maps a control's name to a `Lever(kind,
  baseline)`: `kind` one of `LeverKind.PRICE`, `PROMOTION` and `ORGANIC`, and `baseline` the
  observed least (`Baseline.MIN`, the default), the observed most (`Baseline.MAX`) or a level in
  the control's own units. `MediaPanel.counterfactual(name)` reads it, the least or the most over
  every geo's periods that were not filled. `read_panel(..., levers=...)` reports a lever that
  names no control as a `BAD_SPEC` error.
- **A lever's contribution is its coefficient times its departure from the baseline**, `gamma
  (x_t - b)`, and the base takes `gamma b`: the base is the KPI at the counterfactual level. A lever
  enters with no adstock and no curve.
  - The ridge reads the departure in its design: its columns are centred, so the coefficient is
    the one the raw column takes, and the intercept is the KPI at the baseline.
  - The one-geo Bayesian model samples on the raw column, so its parameters, forecasts, health card
    and scenarios read the control as before; its decomposition moves `gamma b` from the lever's
    part to the intercept's. `lever_contributions(fit, panel, geo)` reads each lever's contribution
    over the window at every draw, the draw's coefficient times the window's summed departure.
- **A lever is never planned.** It is a control, not a channel: no `channels()`, no `Readings`, no
  plan reads it. An organic lever is in the decomposition and in no allocation.
- **The engines that cannot read a lever against its level refuse it.** The DLM's coefficient
  moves, so its base would take a moving `gamma_t b`, and its forecasts read a plan's raw values;
  the models over geos standardise over every geo. Each raises, naming the engines that read it.
- **A panel without levers reads as before**, to the bit: the departure from no baseline is the
  column, and the Bayesian decomposition takes the old path.

## Consequences

- *The decomposition* (`tests/test_levers.py`). On 80 weeks of TV and search, a price of 10 or 8
  in a promotion week and newsletters sent, the ridge with the price a lever measured against its
  most returns the coefficients and the fitted KPI of the same fit without the declaration to
  `1e-9`; the price's contribution is its coefficient times its departure from the most, at or
  above 0 in every week, and the intercept moves by the coefficient times the most. The Bayesian
  model's draws and every other term's part are those of the undeclared fit, bit for bit; the
  price's part is the mean coefficient times the departure, and the parts still sum to the fit.
  `lever_contributions` is each draw's coefficient times the window's summed departure, its mean the
  decomposition's sum to `1e-9`.
- *An organic lever* (a newsletter measured against 0) is in both engines' decompositions and in
  no channel, reading or plan: `plan_posterior` spends the whole budget on TV and search.
- *A panel without levers*: the ridge's fit with its tests, the DLM's fit and its decomposition's
  draws, the Bayesian decomposition on three of Stan's recorded worlds and a short Bayesian fit
  hash to the same bits before and after.
- *Mutations*: the ridge reading the raw price, the Bayesian decomposition leaving `gamma b` in the
  lever's part or out of the intercept's, the least and the most swapped, a filled step read into
  the least, and the DLM's refusal dropped each fail a test.
- *The measurement* (`scripts/study_levers.py`, worlds 0 to 19, then 0 to 142). 104 weeks of two
  channels whose carryover and curve are the model's family, a price of 10 and of 8 in a promotion
  week, a price effect of -40 a unit and the price a lever against its most. Each world is fitted
  by the one-geo Bayesian model, 4 chains of 500 draws, and planned for 13 weeks; regret is per
  euro of the budget against `plan_posterior` on the world's own channels, the scorecard's unit,
  whose claim margin is 0.01.

  | world, KPI noise SD | 90 % holds | 95 % holds | regret mean | its 95 % upper end | above 0.01 |
  |---|---|---|---|---|---|
  | promotions apart from the media, 30 | 16 of 20 | 18 of 20 | 0.0224 | 0.0335 | 11 of 20 |
  | promotions apart from the media, 15 | 16 of 20 | 18 of 20 | 0.0080 | 0.0130 | 4 of 20 |
  | promotions with TV, 30 | 15 of 20 | 16 of 20 | 0.0225 | 0.0334 | 11 of 20 |
  | the last, the price left out | | | 0.0269 | 0.0417 | 13 of 20 |

  - Twenty worlds cannot read a 90 % bar: an exact 90 % interval holds the truth in 18 or more
    of 20 with chance 0.68 only. The first row's world was therefore run again on worlds 0 to 142,
    as many as the compute allowed, each world fitted twice, once with the price read as the lever
    and once with it left out (`--arms lever omitted`):

    | worlds 0 to 142, noise 30 | holds the truth | its exact 95 % interval |
    |---|---|---|
    | the lever's 90 % interval | 127 of 143, 0.888 | [0.825, 0.935] |
    | the lever's 95 % interval | 136 of 143, 0.951 | [0.902, 0.980] |
    | least squares on the true columns, 90 % | 127 of 143, 0.888 | [0.825, 0.935] |
    | least squares on the true columns, 95 % | 133 of 143, 0.930 | [0.875, 0.966] |

    | worlds 0 to 142, noise 30 | regret per euro, mean | its 95 % Student-t interval | median |
    |---|---|---|---|
    | the price read as the lever | 0.0214 | [0.0167, 0.0262] | 0.0102 |
    | the price left out | 0.0353 | [0.0265, 0.0442] | 0.0179 |
    | the first less the second, world by world | -0.0139 | [-0.0222, -0.0056] | -0.0031 |

  - **Coverage: met.** The lever's 90 % interval holds the truth in 0.888 of the worlds, and 0.90
    lies inside the exact 95 % interval of that share; its 95 % interval holds it in 0.951. Least
    squares that knows every channel's true column holds it as often, 127 of 143. The lever's bias
    is +12, with a standard error of 30, on truths averaging 4 040. The first twenty worlds' 16 of
    20 was those worlds': least squares misses the same four, 2, 3, 6 and 7.
  - **Regret: not within the margin.** The plan made with the price read as the lever loses 0.0214
    a euro on average, and the whole of its 95 % interval lies above the margin of 0.01; it loses
    more than 0.01 in 72 of the 143 worlds. Reading the price does not cost the plan: the plan that
    leaves it out loses 0.0139 a euro more, [0.0056, 0.0222], world by world, and is the better in
    57 of the 143 worlds and the same in 6. The regret is the media's identification from 104
    weeks at this noise, not the lever's.
  - The fits diverged in 8, 3 and 42 of the first table's rows' 40 000 draws, and in 170 and 126 of
    the two arms' 286 000 draws over the 143 worlds. 38 and 23 of those worlds' fits have an R-hat
    above 1.01, 5 and 2 above 1.05, the most 1.105; the lever's 90 % interval holds the truth in 93
    of the 105 worlds whose R-hat is at most 1.01, and in 34 of the 38 above.
  - These worlds are a measurement, not a family of the bench's scorecard: a family that confounds
    promotions with the media belongs to the bench.
- The DLM, the DLM over geos and the Bayesian model over geos raise on a lever. A panel that
  declares one is fitted by the ridge or the one-geo Bayesian model, or read without the
  declaration.
- `Lever`, `LeverKind` and `Baseline` are saved and loaded with the panel (ADR 0062).

## Alternatives considered

- **Centre the lever in the Bayesian model's design too.** Rejected: the intercept would then mean
  the KPI at the baseline, and every reader of the raw parameters, the forecast, the health card,
  the scenarios, would have to read the baseline with it; the decomposition alone needs it.
- **A baseline per geo.** Rejected for now: a counterfactual is one policy, "the price at its
  highest", and a level given in the control's units covers a market whose geos differ.
- **Adstock and a curve on a lever by default.** Rejected: a price moves sales in the week it is
  charged, as far as the panel can tell, and a curve on a column that crosses its baseline has no
  reading. A carried-over organic channel is a channel of its own, not a lever.
- **Read a lever in the DLM against 0 and say so.** Rejected: the panel says what the lever is
  measured against, and a decomposition that ignored it would read the same name two ways.

## Amended 2026-11-07: where the regret comes from

The regret of the plan made with the price read as the lever, 0.0214 a euro on worlds 0 to 142 at
noise 30, was taken apart on the same worlds, seeds and fits: each refit drew the recorded regret,
R-hat and divergences to the bit. `scripts/study_levers.py` now writes the parts with each record
and `--summary` prints them. With two channels and a budget the plan is one number, TV's spend a
week, and each channel's worth is its own, so each part is read on 1 001 splits of the week's
budget.

| worlds 0 to 142, noise 30 | regret per euro, mean | its 95 % Student-t interval |
|---|---|---|
| the price read as the lever, as measured | 0.0214 | [0.0167, 0.0262] |
| the same, TV's draws set to the world's values | 0.0168 | [0.0130, 0.0205] |
| the same, search's draws set to the world's values | 0.0124 | [0.0094, 0.0153] |
| the same, each channel's range what the history spent a week | 0.0058 | [0.0030, 0.0086] |
| the posterior's own expected regret of its plan | 0.0445 | [0.0407, 0.0483] |
| least squares told every carryover and curve | 0.0143 | [0.0104, 0.0182] |
| the history's split kept | 0.1490 | [0.1435, 0.1546] |

- **No defect.** The world's regret read again by `chc.response` on the whole series, apart from
  the planner's code, is the recorded one to `3e-15`. `plan_posterior`'s plan is within `5.2e-7` a
  euro of the best split of the posterior's expected worth, and it spends the budget to `1.5e-11`.
  A lever is in no `Readings`: world 75 fitted without the declaration draws the same parameters
  and the same plan, to the bit. At a tenth of the noise, 3, worlds 0 to 19 lose 0.0022 [0.0012,
  0.0031], within the margin; TV's error alone there costs 0.0001, so no unit or carryover
  mismatch is left in TV's reading.
- **Each channel's error alone is above the margin.** With TV's draws the world's, search's error
  is what is left; with search's the world's, TV's.
- **Most of it is where the history never spent.** The best plan gives TV 61 % to 100 % of the
  budget and cuts search below the least it spent in any week of the history in 140 of the 143
  worlds; search's history is a random walk about 100, its least a week 73 at the median world.
  With each channel's range what the history spent, its least to its most, the plan loses 0.0058
  against the best plan within those ranges, and holding them costs the best plan 0.0393 [0.0345,
  0.0442]. At noise 3 the plan within those ranges loses nothing, and all that is left is
  search's error below them, 0.0021.
- **It leans one way.** The plan spends less on TV than the best plan in 104 of the 143 worlds, by
  12.1 a week on average (standard error 1.9).
- **The posterior does not understate it.** Its own expected regret is twice the world's, and the
  world's regret is above 0.9 of the draws' in 8 of the 143 worlds, against 14 for a calibrated
  posterior (exact binomial p 0.09).
- **What these worlds support.** Least squares that knows every carryover and curve, estimating
  the two ceilings and the price, loses 0.0143 a euro with its whole interval above 0.01: on these
  worlds at noise 30 a plan within 0.01 on average is out of reach even with the curves known.
  The Bayesian plan loses 0.0071 [0.0033, 0.0110] more than that world by world, and the plan
  kept at the history's split loses 0.149. The margin these worlds support is about 0.02 with the
  curves known and 0.03 for this plan. The criterion's margin, 0.01, and its ranges, 0 to three
  times each channel's mean, are not changed here.
- The bench now holds a family whose promotions follow the media, family 20 (`promotions`), on
  Track M v2's worlds.

## Amended 2026-11-07: the DLM reads a lever

The DLM's refusal is withdrawn; the models over geos still refuse.

- **The DLM reads a lever as the one-geo Bayesian model does: on its raw column, the declaration
  moving the decomposition alone.** Its filter, its smoother, its signed draws, its tests'
  posterior and its forecasts' locations and scales are those of the same fit without the
  declaration. Its decomposition reads the lever against its level `b`: the lever's contribution
  is `gamma_t (x_t - b)` and the base takes `gamma_t b`, at each period, with a walking coefficient
  as with a still one. The base is then the KPI at the counterfactual level at every period, the
  reading the Bayesian model gives with its one coefficient.
- **`DlmFit.baselines`** holds each lever's level, so that a forecast reads a planned lever the same
  way (`forecast_dlm`), as does every draw of `decompose_dlm`. It is empty without levers, and a
  file written before it loads with it empty.
- **A DLM without a trend refuses a lever**: it has no base to hold `gamma_t b`.
- The objection above, that the base would take a moving `gamma_t b`, is the reading itself: what
  the KPI would be at the counterfactual level moves as the coefficient does.

Consequences:

- *A panel without levers*: the DLM's decomposition, its draws and its forecast, unsigned and
  signed, and a fit with controls among its effects, hash to the same bits before and after.
- *The decomposition* (`tests/test_levers.py`). On the 80 weeks above, with a newsletter whose
  coefficient walks, the states are the undeclared fit's to the bit; the price's contribution is
  the smoothed coefficient times its departure from its most, at or above 0 in every week; the
  base moves by the coefficient times the most; the fitted KPI is the same to `1e-12`. A
  forecast's and each draw's decomposition move the same way, and a fit's levels are saved and
  loaded with it.
- *Coverage.* Worlds of 104 weeks: a level that walks by 3 a week from 600, TV, a price of 10 or 8
  in a quarter of the weeks against its most, an effect of -40 a unit, noise of SD 20; the DLM of
  a level at discount 0.95 and two still coefficients. The central 90 % interval of the price's
  contribution over the weeks, 1000 draws of `decompose_dlm`, holds the truth in 182 of worlds 0
  to 199 (0.91) and in 37 of worlds 0 to 39, where the test asks for 30 or more, which a
  calibrated interval misses with chance 0.0015. Read against 0, the price not declared, it holds
  the truth in none of the 40.
- *Mutations*: the lever's part left against 0 in the fit, in a forecast or in the draws, the base
  left without `gamma_t b` in either, and the refusal without a trend dropped each fail a test.

