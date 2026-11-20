# 0027. A forecast drawn, its signs held over the horizon

Status: accepted, 2026-10-05

## Context

`forecast_dlm` gives a fit's forecast as each period's Student t, exact for the model, and refused
two kinds of fit. A signed effect's future coefficients keep their sign, which a Student t does
not. A transformed effect's future column is the plan's spend carried over from the panel's, which
a fit did not hold. The scorecard's DLM arm fits signed, transformed effects, so its records gave no
forecast, and the scorecard's prediction axis, the CRPS of draws of the quarter's weekly sales at
the status quo, read n/a for it beside every tool that draws one.

## Decision

- **A fit holds its transformed effects' spend** (`DlmFit.spend`), as the panel gave it.
  `forecast_dlm` takes a transformed effect's plan as its spend: the column over the horizon is the
  transform's column of the history's spend followed by the plan's, its last periods, so the
  history carries into the first periods of the horizon as it does in the model. A plan that
  spends below 0 is refused.
- **`forecast_draws(fit, horizon, plan, draws, seed)`**: draws of the KPI over the horizon,
  `(draws, horizon)`, beside `acceptance`.
  - Each draw starts from the states and the variance at the last period given every period: the
    filter's normal-gamma posterior there, or a signed fit's own draws of the posterior truncated to
    the signs, taken with replacement.
  - It is carried through the horizon by the model's evolution, the first step's evolution
    variance held for every later step as `forecast_dlm` holds it, and the KPI is drawn about each
    period's mean at the draw's variance. Without signs the draws follow `forecast_dlm`'s Student
    t.
  - The evolution is `chc.dlm.forecast`'s own, read off it rather than written again: `G`'s column
    `i` is the first step's mean from the unit state `e_i`, and each step's evolution variance is
    the difference of its state covariance and the last one's evolved, symmetrised as `forecast`
    symmetrises it, so a coordinate the model holds reads exactly 0.
  - **A signed fit's path is kept only where every signed coefficient keeps its sign at every
    period forecast.** Over the history and the horizon the model is a Gaussian prior over paths
    conditioned on the signs at every period, so its forecast given the history weighs each start
    by the chance its path keeps the signs, and keeps the path given that it does: rejection with
    the posterior's draws as the proposal is that law exactly. `acceptance`, the share of paths
    kept, says how hard the signs bind over the horizon.
  - A start meets the signs, and each signed coefficient's path is a walk of symmetric steps from
    there, so it keeps its sign over `h` periods with probability at least `C(2h, h) / 4^h`
    (Sparre Andersen), 0.155 at 13: with one signed effect a path kept costs at most 6.5 drawn.
    Several keep their signs together with a chance their steps' correlation sets, at least
    `0.155^k` for `k` whose steps are independent, 269 paths drawn a path kept for three; it is
    above 0 for any start inside the signs, so the loop ends. Paths are drawn 8 192 at a time,
    which at that least keeps, on average, the 13 draws a spec of the arm's 32 takes in one batch.
- **`DlmChoice.forecast`**: the draws spread over the fitted specs as `readings` spreads its own,
  each spec's from a stream of the seed and its index apart from those `readings` takes for the
  same seed (numpy pads a seed with zeros, so a trailing 1 marks the forecast's).
- **The scorecard's DLM arm records the forecast** under `dlm mixture`, as it records the response:
  400 draws of the quarter's weekly sales, every channel at the status quo and the controls at the
  quarter's; the other variants name where it is. The share of paths kept is a flag.
- Rejected:
  - *The quarter as missing observations, sampled again by `chc.dlm.constrained_sample`.* It is
    the model's own law and needs no new numerics, but a missing step compounds the discount, so a
    coefficient's variance grows as `discount ** -h`, 18-fold over 13 weeks at 0.80 where holding
    the first step's grows it 4.25-fold, and `forecast_dlm` holds it; and it runs the sampler again
    for every fitted spec, the arm's dearest step, and changes the history's draws its plan reads.
  - *Drawing each start's path again until it keeps its signs.* That weighs every start alike: the
    forecast of a model whose signs bind over the history alone. A week on, on the signs' test
    world, it lies 0.028 from the model's law at its percentiles, where the draws lie 0.004.
  - *Truncating each step to the signs.* The same weighting, and another law over the path.
  - *Ignoring the signs over the horizon.* The coefficients then cross 0, which is why
    `forecast_dlm` refuses a signed fit.

## Consequences

- *The oracles*, each a test:
  - without signs, each of 13 weeks' 20 000 draws passes a Kolmogorov-Smirnov test against
    `forecast_dlm`'s Student t at 0.001 over the weeks; at 200 000 draws the weeks' variances
    held to 0.5 %;
  - a week on, with the signed coefficient carrying the forecast, the draws lie within 0.004 of
    the model's law, integrated from the fit's draws, at its 99 percentiles, inside the 99 %
    Kolmogorov-Smirnov bound of 0.008, and every start weighed alike lies 0.028 away;
  - over 13 weeks every path kept keeps its sign, and 0.64 of those drawn were kept, above 0.155;
  - a transformed effect's planned column is the history's spend carried over by the transform's
    definition, to 1e-12: with television dark over the horizon, the column is the carryover
    alone, where a column of the plan alone would be 0;
  - two kept specs alike draw distinct forecasts, and no spec's draws meet those its readings'
    stream or their spread would give.
- *Mutations*: twelve, each caught by the tests above:
  - every start weighed alike, its path drawn again until it keeps its signs;
  - the discount compounded rather than held;
  - no observation noise;
  - the last period's covariance not scaled to the draw's variance;
  - the sign read off the wrong coordinate;
  - the history's spend not carried over;
  - the forecast's streams the readings';
  - a signed start at the first period rather than the last;
  - the acceptance counted on the draws kept rather than the paths drawn;
  - the evolution variance not in units of the variance;
  - the evolution transposed;
  - the plan's columns left out of the design.
- *Cost.* On Track M v2's world 905 the arm's 32 fitted specs, every one signed, kept 0.917 of
  the paths drawn over the quarter's 13 weeks, one batch each: 262 144 paths of 13 steps for the
  400 draws, each step one product with the evolution over the spec's states, where a batch sized
  to the acceptance would draw some 440. The batch is kept: the forecast is a small part of a fit
  that samples every spec's posterior, and a fixed batch makes the acceptance's error known.
- The draws are joint over the periods, as a tool's posterior predictive draws are; the
  scorecard's CRPS reads each period's.
- The static Bayesian model still forecasts nothing; its records say so.
