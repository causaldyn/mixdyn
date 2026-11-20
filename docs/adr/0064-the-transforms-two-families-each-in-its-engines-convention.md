# 0064. The transforms: two families, each in its engine's convention

Status: accepted, 2026-11-06

## Context

Every engine reads a channel's spend through a carryover and a curve before a coefficient
multiplies it, and every plan reads the same kernel and curve as a `chc.response.Channel`. The
definitions are spread over the records of the engines that read them: Robyn's transforms in
ADR 0008's ridge engine, the Bayesian model's kernel and Hill in ADR 0010, its `tanh` curve in
ADR 0011, and the same family held as a value in ADR 0020. None states in one place what
`mixdyn.transforms` computes: which kernel is normalised and which is not, what the
half-saturation point is measured in, and which parameter values each accepts. A model ported from
another tool reads the same only where those conventions are the tool's, so they are recorded
here as the code has them.

## Decision

- **Two families, each another tool's convention, never mixed in one channel.** Robyn's
  (`mixdyn.transforms.media`) for the ridge engine; the Bayesian model's
  (`mixdyn.transforms.saturation`), which the DLM over media and the planners also read. Both
  build their kernels and curves from `chc.response`, so a fitted channel is the channel a planner
  plans on. A spend series is a nonempty `(T,)` vector of finite values at or above 0, nothing
  spent before its first period; anything else is refused.
- **Robyn's family**, as Robyn 3.12 computes it (Runge et al. 2024):
  - `Geometric(theta)`: `a_t = s_t + theta a_(t-1)`, an unnormalised kernel `w_l = theta^l`, so a
    constant spend `s` carries to `s / (1 - theta)`; `0 <= theta < 1`.
  - `WeibullCdf(shape, scale)`: `w_0 = 1` and `w_l = prod_(j=1..l) (1 - F(j))`, `F` the Weibull
    CDF of `shape` and `scale`, the share of the carryover that survives each lag, unnormalised;
    `shape` and `scale` positive and finite, the scale in periods. The kernel is as long as the
    series.
  - `robyn_weibull(shape, scale, length)` maps Robyn's parameters onto it: Robyn reads the scale
    as a share of the series' `T` periods, `round(quantile(1:T, scale))`, R's type-7 quantile
    rounded half to even, and a shape or a scale of 0 as no carryover, `Geometric(0.0)`; it takes
    `shape >= 0`, `scale` in `[0, 1]` and `length >= 1`.
  - `Hill(alpha, gamma)`: `x^alpha / (x^alpha + K^alpha)`, `K = gamma max a` over the adstocked
    values of the window the model is fitted on (Robyn 3.12; earlier Robyn releases put `K` gamma
    of the way from the window's least to its most); `alpha > 0`, `0 < gamma <= 1`.
    `half_saturation(hill, window)` computes `K` and refuses a window dark throughout, whose `K`
    would be 0.
  - `carry_over` and `saturate` compute them in NumPy for a fit's inner loop; `channel(adstock,
    hill, half, coefficient, length)` is the same channel in `chc.response`'s terms, its kernel
    `length` periods long and unnormalised, its curve `chc.response.Hill(K, alpha)`.
  - Robyn's Weibull PDF kernel, min-max scaled to `[0, 1]`, is not offered (ADR 0008): the scaling
    sets the smallest weight to 0, and which weight that is depends on the series' length.
- **The Bayesian family** (ADRs 0010, 0011, 0020), held as a value:
  `Transform(retention, half_saturation, shape, length)`.
  - The kernel is normalised: `w_l = r^l / sum_(j<L) r^j` for `l < L`, `L = length` periods, so a
    spend held at `s` for `L` periods carries to `s`; `0 <= r <= 1`, both ends allowed (`r = 1`
    averages the last `L` periods, `r = 0` carries nothing over); `length` a whole number of
    periods, at least 1, a boolean refused.
  - Two shapes, each half its ceiling at `K = half_saturation`, positive and finite, in the
    spend's currency: `TanhShape()`, `tanh(atanh(1/2) a / K)`, PyMC-Marketing's logistic
    saturation `tanh(lam a / 2)` with `K = ln 3 / lam`, concave from no spend and without a slope;
    and `HillShape(slope)`, `a^n / (a^n + K^n)`, `n` positive and finite, S-shaped where `n`
    passes 1. One `K` in both, so that a prior on it states one belief whichever family is fitted
    (ADR 0011).
  - `Transform.kernel()`, `curve()`, `channel(coefficient)` and `column(spend)` build
    `chc.response`'s `GeometricAdstock(r, length=L, normalized=True)` and `Tanh(K / atanh(1/2))`
    or `Hill(K, n)` on demand. A transform holds numbers, not modules, so a spec that carries it
    compares and hashes as a value (ADR 0020).
  - Each refuses JAX in single precision: a curve built in single keeps its rounded parameters and
    its column moves by a part in `1e7`.
- **The Bayesian engines read the family on their draws, not through `Transform`.** The one-geo
  model, the model over geos and the nested model, and the readers of their draws in the
  decisions, the experiments, the health card and the sensitivity bounds, build the same normalised
  kernel and `chc.response.Tanh(K / atanh(1/2))` or `chc.response.Hill(K, n)` per draw, the
  half-saturation point drawn in units of the channel's mean positive spend and carried in its
  currency (ADR 0010). `transforms_from_bayes(fit)` reads a fit's medians as `Transform`s, and its
  columns are the fit's channels' to `1e-15`.

## Consequences

- Robyn's `K` moves with the window it is read on; the Bayesian `K` is a number in the currency. A
  ridge fit refitted on another window keeps `gamma` and moves `K`; a Bayesian fit's channel keeps
  its `K` wherever it is read.
- A currency moves nothing in Robyn's family once saturated, since `K` scales with the spend; the
  Bayesian family's `K` is in the currency, so a model whose spend is rescaled rescales `K` with it.
- A kernel kept in periods moves no earlier value when the series grows; Robyn's share of the
  series moves them all, which is why the Weibull scale is held in periods and mapped only at the
  boundary.
- The constant `atanh(1/2)` is written in `mixdyn.transforms.saturation`,
  `mixdyn.models.bayes_mmm` and `mixdyn.inference.sensitivity`; the tests below hold the family's
  readers to one another, not to one constant.
- The tests that hold the definitions:
  - `tests/test_transforms.py` against Robyn 3.12's own numbers on two spend series
    (`tests/fixtures/oracles/robyn_transforms.json`): `test_geometric_carryover_is_robyn_s` to
    `1e-15` of the largest value,
    `test_weibull_carryover_is_robyn_s_with_its_scale_a_share_of_the_series` to `1e-14`,
    `test_the_hill_is_robyn_s_on_the_window_and_at_other_points` and
    `test_the_hill_is_robyn_s_on_a_window_with_a_dark_period` to `1e-15`, and
    `test_robyn_s_scale_rounds_half_to_even_and_a_zero_carries_nothing`;
    `test_chc_s_channel_is_the_numpy_path` (`chc.response`'s channel is the NumPy path to
    `1e-14`), `test_a_currency_moves_nothing_once_saturated`,
    `test_a_kernel_in_periods_moves_no_earlier_value_when_the_series_grows`,
    `test_robyn_s_share_moves_earlier_values_when_the_series_grows` and
    `test_what_it_cannot_transform_is_refused` (every range above).
  - `tests/test_dlm_media.py`: `test_a_transform_s_column_is_its_kernel_and_curve_by_definition`
    (the column against the definition in NumPy, to `1e-13`, at retentions 0 and 1 among others),
    `test_each_curve_is_half_its_ceiling_at_the_half_saturation_point`,
    `test_a_transform_needs_double_precision`,
    `test_the_first_stage_reads_the_bayesian_fit_s_curves_at_its_medians` and
    `test_what_is_not_a_transform_or_a_planner_s_reading_is_refused`.
  - The Bayesian models' own kernels and curves are held by their Stan oracles (ADRs 0010, 0011,
    0013, 0033): each model's log density and gradient are Stan's to the precision those records
    state, the Stan program writing the kernel and the curve a second time.

## Alternatives considered

- **One family for every engine.** Not done: the ridge engine is to read as Robyn reads a model
  ported from it, with `gamma` and an unnormalised kernel, and the Bayesian engines are to state
  priors on a return and a half-saturation point in the currency, as Meridian and PyMC-Marketing
  do. Each convention is another tool's, and the code keeps both rather than translate one into
  the other.
- **Other kernels and curves.** None is coded: no kernel with a delayed peak, no Weibull PDF
  kernel (above), no curve but these three. A new one enters as a new member of its family's
  union, with an oracle of the tool whose convention it follows.
