# 0076. A return prior that may be 0

Status: proposed, 2026-11-07; adds `HalfNormalRoi` to ADR 0010's return priors, opt-in, a channel
at a time; the model over geos refuses it; no default and no existing output changes

## Context

ADR 0010's model does not sample a channel's coefficient: it sets it from the channel's return
over the window, `roi`, on which it states `LogNormal(mu, sigma)`, Meridian's form, `(0.2, 0.9)` by
default. A log-normal's density vanishes at 0 faster than any power of the return (Maxima,
`validation/half_normal_roi.mac`, step 3), so where the data read no return, the posterior still
cannot reach 0: its interval sits above 0 at a distance the prior sets.

ADR 0014 found this twice in the nested model. Its validation read television's direct return,
0 in half its worlds, never covered by an interval, and the total 0.22 high. Its amendment of
2026-11-06 read television's long-term share, the part of its return that runs through the
searches, on worlds where television does not build the searches at all: the share's 90 %
interval stayed off 0 on 10 worlds of 10, its lower ends 0.040 to 0.110, because stage one's
log-normal kept television's return on the searches off 0 and `gamma`, which the data read near 2,
carried it into the share. The criterion set for that share is that it covers 0 on at least 90 %
of such worlds.

## Decision

- **`HalfNormalRoi(scale, counts="window")`** (`mixdyn.models.bayes_mmm`): a channel's return is
  `HalfNormal(scale)`, in the KPI's units a currency unit, counting what `RoiPrior.counts` names.
  Its density is greatest at 0. `BayesMedia.roi` is a `ReturnPrior`, `RoiPrior | HalfNormalRoi`,
  `RoiPrior()` by default.
- **On the sampler's coordinate.** The sampler moves `u = log roi` for every channel, so the
  model's other coordinates and their transforms are unchanged. The prior's term is the
  half-normal's log density at `exp(u)` plus `u`, the log of the Jacobian:
  `log 2 - log s - log(2 pi)/2 - exp(2 u)/(2 s^2) + u`, which integrates to 1 over the line, its
  gradient `1 - exp(2 u)/s^2` (Maxima, steps 1 and 2). Its left tail falls as `exp(u)`: a
  return's posterior reaches 0 where the data put it there.
- **The default scale, 1.81085**, is `exp(0.2) / Phi^-1(3/4)`, so the half-normal's median is
  `RoiPrior()`'s, `exp(0.2)` (Maxima, step 4). Choosing it moves the prior's mass towards 0 and its
  tail in, not its centre: 4.4 % of its mass lies below 0.1, against 0.27 % of the log-normal's,
  and its 95th percentile is 3.55 against 5.37 (Maxima, step 3). PyMC-Marketing's priors on a
  channel's coefficient are half-normal.
- **A channel at a time.** One model can hold half-normal and log-normal returns side by side. A
  model with no half-normal return computes exactly what it did before: the same expressions, in
  the same order.
- **The model over geos refuses it.** `BayesGeoSpec` pools each channel's log return over the geos
  about the national one; a half-normal on the return has no such centre. It names the channels.
- **The nested model reads `HalfNormalRoi()` in stage one at the mediator's scale**, its scale
  times the mediator's mean over the KPI's, as it reads `RoiPrior()` (ADR 0014, amended
  2026-11-07); a scale stated is read as stated, in the mediator's units a currency unit.
- **The health card's prior draws** draw it as the absolute value of a normal draw of SD `scale`,
  from the stream a log-normal draw takes, so a model without one draws what it did. The factsheet
  states it as `HalfNormal(scale)`, stated by the caller. The file format saves and loads it.

## Consequences

- *Where television does not build the brand* (`validation/long_term_null.py`, its second table
  in `validation/long_term_null.md`, `tests/test_long_term_null.py`; ADR 0014's long-term null
  worlds, seeds `SeedSequence(20261106).spawn(100)`, worlds 0 to 99; two chains of 300 warm-up
  iterations and 200 draws; television's return on the searches `HalfNormalRoi()`, its scale near
  0.29 searches a euro at the searches' scale). The share's 90 % interval covers 0, its lower end below
  0.02, on 87 of 100 worlds: 0.87, its Monte Carlo standard error 0.034, its 95 % Clopper-Pearson
  interval [0.788, 0.929] (Maxima's beta quantiles agree). The harness's log-normal,
  `LogNormal(0, 1)`, covered it on 0 of 10. The criterion set for it, at least 90 %, met where that
  interval holds 0.9 or lies above it, as ADR 0014 reads its own, is met; the interval does not
  rule out a rate down to 0.788 either, and a rate of 0.9 gives at most 87 of 100 with probability
  0.20 (Maxima). The first 20 worlds gave 15, [0.509, 0.913], and the run was extended to 100 on
  the same seeds and settings, the 20 kept, to tell 0.75 from 0.9.
  - *The 13 misses*, lower ends 0.021 to 0.116. On the five among the first 20, refitted, stage
    one's median return on the searches is 0.059 to 0.18 a euro and its 5 % quantile 0.009 to
    0.044; on the 15 that cover, 0.016 to 0.059 and 0.001 to 0.007. With `gamma` near 2 and the
    total near 1, the share's lower end passes 0.02 where that quantile passes about 0.01. Where the
    data read a return off 0 by chance, a density at 0 that is finite does not pull it back: the
    misses are the data's, and a prior that covered 0 on them would read a return as 0 harder than
    the data do, the third alternative below.
- *Where sales do not read the brand*, the same worlds' other kind: the share covers 0 on 18 of 20,
  0.90, standard error 0.067, [0.683, 0.988], against the log-normal's 17 of 20. No fit of either
  kind had a divergent transition.
- *Where television builds the mediator* (ten worlds of `tests/test_nested.py`'s design, data
  seeds 20261002 to 20261011; the world's return 0.5 searches a euro): the half-normal held
  television's total return on sales on 10 of 10 worlds, its medians 0.50 to 1.12 against the
  world's 1, and its return on the searches at medians 0.26 to 0.56. On the search-demand world of
  ADR 0074 (seeds 0 to 9, two chains of 500 and 500), it held the total on 8 of 10, but stage one
  did not converge on two of them, 124 and 23 divergent transitions and an R-hat of 1.37 and 1.35,
  its intervals of the queries a euro [12, 311] and [4.6, 360] against the world's 51 and 50. The
  log-normal at the queries' scale had 0 and 41 divergent transitions there, R-hat 1.01 and 1.04,
  and held the total on 7 of 10. Both stage ones lacked a trend, and television's carryover at a
  decay near 1 took the queries' drift. With a trend of 3 bends in stage one, the half-normal had
  no divergent transition on the ten worlds, R-hat at most 1.022, and held the total on 9 of 10
  (ADR 0014, amended 2026-11-07). A half-normal is for a channel that may not move the mediator,
  stated where that is the question, and its fit is read with the health card.
- A model with no half-normal return computes what it did, the same expressions in the same
  order: on the two worlds of the engine's Stan reference (`tests/test_bayes_mmm.py`, `"normal"`
  and `"tanh"`), its log density and gradient at ten points and a fit of two chains of 50 warm-up
  iterations and 20 draws equal those before this change, array by array, to the bit.

## Alternatives considered

- **A spike and slab**: the return 0 with some probability, else log-normal. The engine's NUTS
  moves on continuous coordinates; an indicator per channel would have to be summed out, two
  terms a channel and `2^M` mixture components a model, and the sampler would meet the mixture's
  funnel at the spike. Not built.
- **A wider log-normal.** Its density still vanishes at 0; a wider one pushes mass to large
  returns as much as to small ones.
- **A density that grows without bound at 0**, a Gamma of shape below 1 or a horseshoe. It reads
  a return as 0 harder than the data do, where a channel works but weakly, and its log density's
  slope near 0 is unbounded on the sampler's coordinate. A half-normal's density at 0 is finite:
  the posterior reaches 0 as far as the data allow and no further.
- **An exponential.** The same density at 0 up to its scale, a heavier tail than the half-normal
  at the same median; PyMC-Marketing's family and a lighter tail were preferred.
- **Half-normal as the default.** It would move every fit's numbers, and Meridian's log-normal is
  the prior its users calibrate against. It stays opt-in.
