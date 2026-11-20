# 0066. The DLM core: `chc.dlm`'s blocks on the panel's calendar, and its default prior

Status: accepted, 2026-11-06

## Context

The dynamic linear model (`mixdyn.models.dlm`) is the engine whose coefficients move. Later records
build on it: ADR 0020 gives an effect a carryover and a curve, ADR 0022 draws a kept spec's signed
chains again, ADR 0026 gives each effect its own discount, ADR 0027 draws a signed fit's forecast,
ADR 0038 reads a lift test into it, and ADR 0007 fits it over geos with a prior "as `vague_prior`
does". The model those records extend was never recorded itself: how the state is laid out, how
the calendar's gaps enter, what an idle effect does, which prior a fit starts from, how an effect
is held to a sign, and how a fit splits into its parts and forecasts. Each is a choice a fit's
numbers depend on, written in the code's docstrings and in the commit that introduced the model
(2026-10-01).

## Decision

- **The state is `chc.dlm`'s structural blocks, in one order** (`DlmSpec.blocks()`): a trend
  (`Trend(order, discount)`, `chc.dlm.Polynomial`, a level or a level and a slope, one discount),
  then one random-walk coefficient for each effect (`Effect(column, discount)`,
  `chc.dlm.Regression`), then a season (`Season(period, harmonics, discount)`, `chc.dlm.Seasonal`,
  harmonics 1 to `harmonics` of a period counted in the panel's periods, 52.18 weeks to a year
  allowed). Trend and season are states, not regressors on a time index and Fourier terms, so a
  missing period evolves them as it evolves every other state. The filter, the smoother, the
  forecast and the likelihood are `chc.dlm`'s, conjugate normal-gamma throughout; `form`
  (additive or multiplicative) and `variance_discount` are its too. An effect named twice is
  refused.
- **Time is the panel's calendar** (ADR 0002). A period without a KPI is a missing step, filtered
  without an update and evolved as any other; a missing value of an effect's column enters as 0.
- **An idle effect is held.** While its column is 0 an effect's coefficient is not discounted
  (`hold_when_idle=True`, the default), so a channel that goes dark does not come back with its
  variance grown by `discount ** -steps`. A forecast holds a coefficient whose planned column is 0
  the same way.
- **The default prior is `vague_prior(panel, geo, spec)`**, weakly informative on the series' own
  scale. Every coordinate's mean is 0 but the level's, which starts at the KPI's mean over its
  observed periods. Each standard deviation is ten times its coordinate's scale in the forecast:
  the KPI's standard deviation `s` for the level and each season coordinate, `s / T` for the
  slope over the calendar's `T` periods, and `s / rms(x)` for an effect, `rms(x)` over the
  observed periods, `s` for a column that is 0 throughout them. The observational variance is
  learned from one degree of freedom at `s^2`. The KPI's mean and standard deviation are the only
  numbers the data supply twice. A geo with fewer than two observed periods, or whose KPI does not
  vary, sets no prior and is refused. A caller's own `chc.dlm.Prior` replaces it.
- **A sign is a truncation, not a clip.** `Effect(sign="positive" | "negative")` holds a
  coefficient to its sign at every period: `fit_dlm` then draws the posterior truncated to the
  signs by `chc.dlm.constrained_sample`, four chains of `draws` after as many of warm-up from
  `seed`, and what the data pull against a sign moves into the other states, the base first. It
  is drawn only for an additive model with a variance discount of 1, and any other spec with a sign
  is refused at construction. Whether the chains mixed is `ConstrainedDraws.mixed`, which a choice
  acts on (ADR 0022) and the diagnostics report (`CheckCode.UNMIXED`).
- **The decomposition is read off the states.** `base` is the trend's level, `seasonality` the
  sum of the season's observed coordinates (the first of each harmonic's pair, and the single
  coordinate of a harmonic at the Nyquist frequency), and each effect's contribution its
  coefficient times its column; `estimated` is their sum. The states are the smoothed means, or
  with a signed effect the mean of the truncated posterior's draws, or with lift tests the
  posterior given them (ADR 0038). A fit without a trend has no base, and one without a season no
  seasonality.
- **A forecast takes the effects' future columns from a plan**, never forecast: spend is a
  decision (`forecast_dlm(fit, horizon, plan)`; a plan names every effect and nothing else). Each
  period's KPI is the model's Student t, split into the same parts. The first step's evolution
  variance is held for every later step, so a state's variance grows linearly with the horizon,
  not as `discount ** -horizon`. A signed fit's forecast is drawn instead (ADR 0027).
- **The fit's likelihood is the filter's one-step log-likelihood**, plus the tests' log density
  given the KPI where the fit read tests (`DlmFit.log_likelihood`), the number `choose_dlm` ranks
  discount specs by.

## Consequences

- A fit is equivariant to the KPI's unit, moves only its base when a constant is added to the KPI,
  and leaves every contribution alone when a column is rescaled: the default prior reads each scale
  off the data it fits. On the tests' world of 104 weeks with a trend, a 52-week season, two
  channels and two controls, each equivariance holds to `4.9e-14`.
- A held coefficient keeps its variance through a dark spell. On a daily world whose holiday
  column first comes on at day 326, held, the coefficient's variance over the first 325 days never
  grows; discounted at 0.95 every day, it grows by `0.95 ** -324`, about `1.65e7`-fold.
- A sign the data contradict moves the channel's part into the base, and the fit still adds up to
  the KPI.
- The model over geos (ADR 0007) shares the blocks and the prior's rule on each geo's scale, but
  draws no sign and reads no transform.
- The tests that hold the core:
  - `tests/test_dlm.py`: `test_a_discounted_level_settles_at_its_maxima_steady_state` (a level
    discounted at `delta` settles at `C = V (1 - delta)`, against causal-hybrid-control's Maxima
    derivation, to `1e-12`), `test_an_idle_effect_is_held_while_its_column_is_zero`,
    `test_the_default_prior_makes_the_fit_equivariant_to_the_kpi_unit`,
    `test_the_default_prior_moves_only_the_base_when_the_kpi_shifts`,
    `test_the_default_prior_leaves_contributions_alone_when_a_column_is_rescaled`,
    `test_the_default_prior_lets_the_slope_move_the_level_as_far_as_the_level_spreads`,
    `test_a_kpi_that_does_not_vary_sets_no_prior`, `test_an_effect_named_twice_is_refused` and
    `test_an_effect_the_panel_lacks_is_refused_by_name`;
  - `tests/test_dlm_media.py`: `test_at_discount_one_the_filter_is_the_batch_conjugate_regression`,
    on a world with a missing KPI week: at discount 1 the filter at the last week is the batch
    normal-gamma regression of the observed weeks alone, its degrees of freedom the prior's plus
    the observed weeks' count;
  - `tests/test_dlm_signs.py`: `test_a_sign_the_data_contradict_moves_the_channel_into_the_base`,
    `test_a_signed_fit_still_adds_up_to_the_kpi`,
    `test_every_draw_holds_the_sign_and_the_point_decomposition_is_their_mean`,
    `test_only_the_signed_effect_is_held_to_its_sign` and
    `test_a_signed_effect_in_a_model_the_sampler_cannot_draw_is_refused`;
  - `tests/test_dlm_forecast.py`: `test_the_forecast_variance_grows_linearly_with_the_horizon`,
    `test_the_forecast_splits_into_its_parts`, `test_the_forecast_carries_the_season`,
    `test_the_forecast_carries_the_planned_spend` and
    `test_a_plan_must_name_every_effect_and_nothing_else`;
  - `tests/test_dlm_decomposition.py`: `test_the_draws_average_to_the_smoothed_decomposition`.

## Alternatives considered

- **Trend and season as regressors** on a time index and Fourier terms. Not taken when the
  model was written: as states, a missing period evolves them as it evolves every other state.
- **Every coefficient discounted at every step.** Not the default: a coefficient idle for `k`
  steps would come back with its variance grown by `discount ** -k`, as the holiday above shows.
  `hold_when_idle=False` keeps it available.
- **A prior from a regression on the series** (all of it, or a first segment), or a level that
  starts at 0. Not the default: a regression's prior reads the same periods the filter reads
  again, and a level that starts at 0 makes the fit depend on where the KPI's zero lies, which the
  shift equivariance above rules out.
  `scripts/study_dlm_prior.py` compares the default with these and with the idle hold dropped, on
  held-out log scores and coverage; no run of it is recorded in this tree.
- **A sign by clipping**, the wrong-signed contribution of a free fit set to 0. Not taken: it
  leaves the free fit's base, and the parts then miss the KPI's average by nearly the channel's
  whole effect, 22.8 on the signs' test world, where the truncated posterior misses it by 0.03
  (`test_a_signed_fit_still_adds_up_to_the_kpi`).
