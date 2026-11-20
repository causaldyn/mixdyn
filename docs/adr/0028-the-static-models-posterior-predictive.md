# 0028. The static model's posterior predictive

Status: accepted, 2026-10-05

## Context

The static Bayesian model (`mixdyn.models.bayes_mmm`) drew its posterior and read each channel's
return and curve from it, but forecast nothing past its window. The scorecard's prediction axis,
the CRPS of draws of the quarter's weekly sales at the status quo, so read n/a for the engine's
static arm beside every tool that draws one.

## Decision

- **`forecast_bayes(fit, horizon, plan, draws, seed)`**: the model's posterior predictive over the
  `horizon` periods after the window, `BayesForecast(dates, mean, kpi)`, each `(draws, horizon)`
  in the panel's units. `plan` gives each channel's spend and each control's value at each period.
  - Each draw is at a draw of the posterior evenly spaced over the chains' from the first to the
    last, as `BayesFit.channel_draws` reads them, so a forecast and readings of one count are at
    the same draws, in order.
  - Its mean is the fitted mean's formula run on: the intercept; the trend's line from the
    window's last bend, the knots still the window's and the columns still in units of its length
    (`LinearTrend.columns(periods, ahead=...)`, `BayesSpec.calendar(periods, ahead=...)`); the
    season's harmonics on their period, counted from the window's first period; each control's
    coefficient times its planned value; and each channel's coefficient times its curve of its
    adstock, read over its spend from the panel's first period to the window's last followed by
    the plan's, through the kernel the fit read. A channel is built in one place for the planner's
    readings and the forecast (`BayesFit._channel`).
  - The KPI is drawn about each draw's mean by the model's noise at that draw: normal at its
    scale, Student-t at its scale and degrees of freedom, or log-normal whose mean is the draw's
    mean.
  - Under log-normal noise a draw whose mean is not above 0 at a period has no KPI there, as the
    model's density has none: the forecast is refused, naming the periods and how many draws.
  - JAX in single precision is refused, as the model refuses it: the channels run through
    `chc.response` in JAX, which would read the draws in single precision without a word.
- **A fit holds its channels' spend** from the panel's first period to the window's last
  (`BayesFit.spend`), which the adstock reads, **and the panel's cadence** (`BayesFit.cadence`),
  which dates the forecast, as `DlmFit` holds both.
- **The scorecard's static arm records the forecast** under `stacked`, as it records the response:
  a draw at each of the stacked posterior's readings, the Hill's first as the readings run, every
  channel at the status quo and the controls at the quarter's, each family's noise from a child of
  the export's seed apart from the fits'. `cvar 0.1` names where it is; an export without the
  quarter's controls has none, and says so.
- Rejected:
  - *The quarter as unobserved periods of a longer window, fitted again.* The model reads no KPI
    where none is observed, but a longer window is another model: the knots spread over it, the
    trend's columns are in units of its length, the controls are standardised over it, and each
    channel's coefficient is set by its return on the window's spend, which would hold the plan's.
  - *New bends in the trend past the window*, Prophet's forecast uncertainty. The model has no
    such bends; its predictive is its own line run on, and a forecast that drew them would be of
    another model.
  - *A predictive at every draw of the posterior.* The planner's readings are a thinned count; a
    forecast of the same count at the same draws keeps the two of one posterior.

## Consequences

- *The oracles*, each a test:
  - the fitted mean's formula, written again in numpy, averaged over every draw at the window's
    periods, is the fit's own posterior mean of its fitted mean to `1e-12`, on a Hill's fit and a
    `tanh`'s whose window starts 24 weeks into the panel; past the window the forecast's mean is
    that formula at each of 37 draws to `1e-12`, the spend before the window carried in;
  - a model of its channels alone forecasts its intercept and `BayesFit.channel_draws`' channels
    run over the spend and the plan's, to `1e-12`;
  - each KPI drawn, standardised at its own draw's noise, passes a Kolmogorov-Smirnov test against
    the noise's law at 0.001: normal; Student-t at the draw's degrees of freedom, 1.5 and 60 in
    turn and read apart; and the log's normal. Each draw's scale is spread at least twentyfold, so
    a KPI drawn at another draw's noise would be seen;
  - `LinearTrend.columns(120, ahead=10)` is the window's 120 rows, then the line on past its last
    bend.
- *Mutations*: twelve, each caught by the tests above:
  - the calendar of a window the horizon longer;
  - the knots spread over the horizon too;
  - the trend's columns in units of the window and the horizon;
  - the history's spend not carried over;
  - the spend held to the panel's end rather than the window's;
  - the first draws rather than those evenly spaced;
  - the controls left out;
  - one noise scale for every draw;
  - one degrees of freedom for every draw;
  - a log-normal about its median rather than its mean;
  - a plan below 0 let through;
  - a mean at 0 let through under log-normal noise.

  One degrees of freedom for every draw survived a first test whose degrees of freedom ran evenly
  from 2.5 to 30. A law drawn at one value between them is too light-tailed for some draws and too
  heavy for others, and pooled together the two errors cancel. The test now gives the draws 1.5 and
  60 in turn and reads each group apart.
- The forecast's draws are joint over the periods; the scorecard's CRPS reads each period's.
