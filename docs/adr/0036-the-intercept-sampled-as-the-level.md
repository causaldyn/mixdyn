# 0036. The intercept sampled as the level

Status: accepted, 2026-10-06

## Context

`test_the_joint_posterior_is_stan_s[both]` failed on main's CI at `607273d`, on runs 37412160008
and 37422226137, bit for bit on both: the nested model's calendar coefficients read an R-hat of
1.08 and a bulk ESS of 31. The runner was an AMD EPYC 7763, x86-64-v3 without AVX-512. On this
machine the test passed on six seeds and on the replayed shard.

On the runner, at seed 1 and the default target 0.8, one chain of four diverged on 397 of its
1 000 draws. Its mean tree depth was 4.2 against the others' 6.1, and its mean log density 515.2
against 517.6: warm-up had left it in a part of the posterior the others did not visit. At target
0.9 that seed diverged nowhere.

The runner only made the fit fail. Here the same fit diverged almost everywhere, and so did Stan's:

| sampler, target | seeds | seeds diverging | chains diverging | divergences | most in a chain |
|---|---|---|---|---|---|
| the engine, 0.8 | 7–30 | 23 of 24 | 67 of 96 | 184 | 9 |
| the engine, 0.9 | 1–22 | 19 of 22 | 36 of 88 | 125 | 42 |
| CmdStan 2.38, 0.8 | 1–40 | 40 of 40 | 110 of 160 | 325 | 23 |

The cause is a ridge. The KPI's mean pins the intercept plus the channels' contribution, and
neither one alone. On the draws of seed 12 at 0.9, the second stage's intercept and the channels'
contribution averaged over the window, `sum_m roi_m spend_m / periods`, correlate at −0.997, the
first stage's at −0.974. The standard deviation of their sum is 0.0009, against the intercept's
0.0115. A diagonal metric cannot follow a direction that runs across two coordinates, so the step
is held to the ridge's width and the trees run deep: steps of 0.035 to 0.072 and mean depths of
5.8 to 6.8 at 0.9. A chain that adapts its step where the ridge bends diverges, or stays there.

ADR 0012 met this ridge at the start of warm-up, a chain whose intercept was negative and whose
first channel stood in for the baseline, and moved the starts. The ridge stayed.

## Decision

- **`BayesModel` samples the level in the intercept's place**: `level = intercept + sum_m
  exp(log_roi_m) window_spend_m / periods`, the coordinate named `level`. `_constrained` maps it
  back, `intercept = level - _media_level(log_roi)`.
- The map from `(log_roi, level)` to `(log_roi, intercept)` is triangular with a unit diagonal.
  Its Jacobian is 1, so the posterior is the same and the density gains no term. The prior
  `intercept ~ N(1, 1)` still reads the intercept.
- Every parameter keeps its name. Draws, diagnostics, fits and plans read `intercept` as before;
  only `coordinates` names `level`.
- `_lift` raises the level. That raises the intercept by as much and keeps every other coordinate.
- The Stan program in `tests/fixtures/oracles/bayes_mmm_stan.json` still samples the intercept.
  The tests map Stan's points by `_with_level`, compare the density there and the gradient by the
  chain rule, so the oracle checks the map as well as the model.
- A test holds the level to its meaning: at Stan's points, the intercept plus the mean over the
  window of the model's own channel terms, to 1e-12.
- The nested model's stages are `BayesModel`s and take it. The model over geos keeps its
  coordinates (Consequences).

## Consequences

- *The nested fit of the failing test*, here, on the seeds above:

  | coordinate, target | seeds | chains diverging | divergences | steps | mean depths | largest R-hat |
  |---|---|---|---|---|---|---|
  | intercept, 0.9 | 1–22 | 36 of 88 | 125 | 0.035–0.072 | 5.75–6.82 | 1.0105 |
  | level, 0.9 | 1–22 | 0 of 88 | 0 | 0.116–0.186 | 4.33–4.96 | 1.0065 |
  | intercept, 0.8 | 7–30 | 67 of 96 | 184 | not recorded | not recorded | 1.0051 |
  | level, 0.8 | 7–30 | 0 of 96 | 0 | 0.135–0.194 | 4.27–4.88 | 1.0058 |

  The steps are about three times as long, and the trees about one level and a half shallower. On
  the draws of seed 12 at 0.9, the level and the channels' contribution correlate at +0.015 in the
  first stage and −0.000 in the second.
- The density's value at a point of the new coordinates is the old one's at the mapped point, so
  every recorded posterior, Stan's 40 000 draws among them, is still the oracle. A saved theta from
  an earlier fit, read in its coordinates, would be read wrongly; no record keeps one.
- *Tests*: the density and its gradient against Stan's at Stan's points, mapped, for the one-stage
  and the nested model; each observation's log likelihood at Stan's recorded draws, mapped; the
  level's meaning at Stan's points; the start's lift and the support's edge on the level.
- *Mutations*, each run on a copy of the tree: 8, of the level's sum (4: the level read as the
  intercept, not averaged over the periods, the channels averaged, the log return for the return),
  the map back and its inverse (3) and the lift (1). Each fails a test. The four of the sum leave
  Stan's density checks green, since a map and its inverse that agree pass them whatever the sum:
  the test of the level's meaning is the one that fails them.
- *The scorecard's static arm on the drift family's pilot*, its 50 drifting worlds at k = 0 fitted
  and planned again at this ADR's commit, everything else as at `607273d`. The fits diverged in 1
  world of 50, once, against 47 of 50 and up to 559 times in 4 000 draws before. Seven of the 50
  plans read a fit past the scorecard's bounds, against 12, each by an R-hat of 1.010–1.028 and
  none by its divergences. The plans did not move: `stacked` planned the same split in 43 of 50
  worlds, and its regret moved by +0.0067 per euro [−0.0098, +0.0232] (paired, 95 %), `cvar 0.1`'s
  by +0.0011 [−0.0027, +0.0048]. The drift run's pre-registration reads this arm.
- *Left*: the model over geos samples each geo's base on the log scale, and its base multiplies
  the calendar, so the same map would carry a Jacobian. Whether its fits meet the ridge is not
  measured yet; it is changed if they do.

## Alternatives considered

- **A higher target.** At 0.9 the runner's seed did not diverge, but here 19 of 22 seeds still did,
  one chain 42 times: a smaller step on the same ridge.
- **A dense metric.** It could learn the ridge's direction, from a warm-up that the ridge itself
  slows. Stan's default, which the oracle runs, is diagonal. Rejected: the map removes the ridge at
  no cost, where a dense metric would have to learn it on every fit.
- **A longer warm-up.** Rejected: the runner's chain adapted to the ridge, and more warm-up gives
  it more time there, not another geometry.
- **The channels' contribution centred in the model itself.** The same posterior, but the
  intercept would change its meaning for every reader of `intercept`. The level changes only the
  sampler's coordinates.
