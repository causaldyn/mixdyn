# 0059. Simulation-based calibration, recorded

Status: accepted, 2026-11-01; amended 2026-11-10: the 200-simulation records replace the
40-simulation ones; every engine passes at its defaults, two centred records fail

## Context

The Bayesian engines are checked against Stan's posteriors on fixed worlds (ADRs 0010, 0013,
0017): on those worlds the engine's sampler draws what Stan draws. That says nothing of whether a
posterior is right over the worlds its prior allows, nor of where the sampler fails between them.
Simulation-based calibration (Talts, Betancourt, Simpson, Vehtari and Gelman 2018) asks that over
the prior: if each world is drawn from the prior and its data from the likelihood, the rank of the
true parameter among the posterior's draws is uniform, and a miscalibrated density, sampler or
start shows as a rank histogram that is not. Modrák, Moon, Kim, Bürkner, Huurre, Faltejsková,
Gelman and Vehtari (2023) give the protocol this ADR follows. No record of it existed in the tree,
and centred geo fits were known to diverge (ADR 0013) without a trace of why.

## Decision

- **The harness** is `validation/sbc.py`:
  `python -m validation.sbc --engine <name> --sims N --seed S --world small|centred-geo
  [--knob NAME=VALUE] [--workers W]`. Each simulation draws a point from the engine's prior on its
  sampling coordinates, simulates the data with the engine's own mean and noise functions
  (`_constrained`, `_terms`, `_mean`), fits them with the engine's sampler at its defaults (four
  chains of 1000 draws after 1000 of warm-up, the engine's own starts), and records each true
  coordinate's rank.
- **On the model's scales.** The engines read the KPI over its mean, so their priors are stated on
  a scale the data set; simulated data would move it, and the fit's posterior would no longer be
  the one the prior and likelihood make. The harness builds each world's model once and replaces
  its KPI by the simulated one. What is calibrated is the density, the sampler and the starts; the
  scaling is a fixed map of the data.
- **The engines and their small worlds**, 52 weeks of spend in flights, two channels, Hill curves,
  normal noise, no calendar or controls:
  - `bayes_mmm`, one geo, chains started by Pathfinder;
  - `bayes_geo`, four geos of unequal populations, non-centred, chains started uniform; its
    `centred-geo` world is the same with both channels centred;
  - `nested`, stage one `tv` on a mediator, stage two `search` and the mediator held at or above 0,
    without lift estimates, so the stages are drawn apart from seeds spawned as `fit_nested`
    spawns them; stage one's simulated mediator is stage two's control;
  - `dlm`, `fit_dlm` with both effects held positive, the part of the DLM a sampler draws
    (`chc.dlm.constrained_sample`), its prior `vague_prior`'s on the world's panel, held fixed.
    Every discount is 1: with a discount below 1 the evolution's variance is the filter's, and the
    prior over a coefficient's path truncated to its sign is not one the harness draws. Its ranks
    are the last step's states and the log variance. `geo_dlm` samples nothing.
- **The thinning.** A rank is read among `L = 63` draws evenly spaced over the 4000, so it takes
  one of 64 values and the thinned draws are close to independent at the engines' ESS.
- **The uniformity test.** Pearson's chi-square over `B` equal bins of the 64 ranks, `B` the most
  of 2, 4, ..., 64 that leaves five simulations a bin (8 at 40 simulations, 32 at 200). A block of
  coordinates (`log_roi`, `deviation`, ...) is read by the least of its coordinates' p-values
  times their number (Bonferroni), which holds whatever their dependence within a simulation.
- **The pass rule**: at 200 simulations, every block's p-value above 0.01. A record of fewer
  simulations is a reading, not a pass.
- **The records** live in `validation/sbc/<engine>-<world>[-<knob>-<value>].json`, each
  simulation's truth, ranks, divergent transitions, the mean of the draws at them, every
  coordinate's R-hat and bulk and tail ESS, and ADR 0035's compute record (a fit's wall, CPU and
  compiling seconds, the process's peak resident set, the least bulk ESS per CPU second, a
  failure's kind); and in the `.md` beside it, the command, the date, each block's p-value, each
  coordinate's rank histogram as text, the divergences and where they fall. The compute is the
  record of a run under its machine's load, for sizing runs, not a timing.
- **The geo model's knobs** (`BayesGeoSpec`), opt-in, each the default unchanged bit for bit:
  `noise_prior="log-normal"`, `sigma ~ LogNormal(log 0.6745, 0.5)`, the half-normal's median held
  away from 0; `base_prior="normal"`, `a_g ~ N(1, 1)` sampled as it is, a base allowed below 0;
  `slope="fixed"`, each Hill's slope 1 and not sampled. Which channels are centred was already a
  choice (`centred`).
- **The CI test** (`tests/test_sbc.py`) runs 12 DLM simulations and asserts the harness runs, its
  ranks are in range and its test does not reject at 0.001: it pins the harness, not calibration.

## Consequences

- *The small worlds at 200 simulations* (seed 0, `B = 32`), no fit failing:

  | engine | least block p | its block | fits with a divergence | fits with R-hat > 1.01 | fits with bulk ESS < 400 |
  |---|---|---|---|---|---|
  | `bayes_mmm` | 0.152 | `log_slope` | 2 of 200 | 0 | 1 |
  | `bayes_geo` | 0.061 | `log_half_saturation` | 6 of 200 | 0 | 0 |
  | `nested` | 0.161 | `two.logit_retention` | 13 of 200 | 2 | 1 |
  | `dlm` | 0.357 | `log_variance` | Gibbs, none | 0 | 0 |

  Every engine passes the rule at its defaults. At 40 simulations `nested`'s stage one read its
  half-saturation at 0.004, its ranks piled at 0, and its noise scale at 0.012; at 200 they read
  0.188 and 0.497: chance, as eighty unrecorded simulations from seed 1 had suggested.
- *The centred geo world* diverges in every fit: 200 of 200, a fit's median 250.5 divergent
  transitions in 4000 and most 1036; 158 fits have an R-hat above 1.01 and 135 a bulk ESS below
  400. Non-centred, on the same designs and draws, 6 of 200 fits diverge, once each.
- *The knobs*, one at a time on the centred world, 200 simulations each from seed 0:

  | knob | fits with a divergence | a fit's median | R-hat > 1.01 | bulk ESS < 400 | least block p |
  |---|---|---|---|---|---|
  | none (both centred) | 200 of 200 | 250.5 | 158 | 135 | 0.023 `log_spread` |
  | `noise_prior="log-normal"` | 200 of 200 | 257 | 156 | 140 | 0.104 `log_spread` |
  | `base_prior="normal"` | 200 of 200 | 283 | 168 | 146 | 0.111 `log_spread` |
  | `slope="fixed"` | 200 of 200 | 250.5 | 161 | 144 | 0.006 `log_spread`, fails |
  | `centred={"tv"}` | 200 of 200 | 224.5 | 123 | 109 | 0.003 `log_spread`, fails |
  | `centred={"search"}` | 200 of 200 | 211 | 119 | 110 | 0.053 `log_spread` |
  | `centred={}` (the small world) | 6 of 200 | 0 | 0 | 0 | 0.061 `log_half_saturation` |

- *The reading.* No prior moves the divergences; centring does, and one centred channel is enough
  for every fit to diverge. The draws at the divergent transitions sit below a centred channel's
  true spread: with `tv` alone centred, 1.04 below `log_spread[tv]` on average (below it in 167 of
  the 200 fits) and 0.03 below `log_spread[search]`'s; with `search` alone, 1.13 below its own (173
  of 200); with both, 0.47 and 0.57 below (139 and 154 of 200). The divergences are the centred
  form's funnel: four geos give each channel three deviations, too few for the data to pin a geo's
  effect, so the posterior of a centred channel's deviations narrows with its spread, and the
  sampler cannot follow the neck. That is ADR 0013's warning, that centring suits a channel whose
  geo effects the data pin, read over the prior: in a world of four geos and 52 weeks no draw of
  it pins them. At 200 simulations the ranks show it too: `log_spread` is the least block of every
  centred record, and two of the six fail the rule, `centred={"tv"}` at 0.003 and
  `slope="fixed"` at 0.006. The flag of 40 simulations, the log-normal noise prior's `deviation` at
  0.004, reads 0.963. No knob is a remedy; the non-centred default passes, and centring stays off
  unless a channel's geo effects are pinned by the data.
- *Recorded 2026-11-10*: the 200-simulation records of every engine and world and of the knob
  table, run with the same commands and `--sims 200` (CHC 1.3.0 and jax 0.12.0, each record's
  `versions`), replace the 40-simulation ones.
- *Not covered*: the knobs on the non-centred world; lift estimates, geo tests and series in the
  likelihood; log-normal and Student-t noise; calendar and controls but the nested stage two's
  mediator; the carried return; a DLM whose discounts are below 1 and its lift tests. The harness
  refuses each it cannot draw.

## Alternatives considered

- **Simulating in the panel's units and fitting with `fit_bayes`.** Rejected: the model rescales
  the KPI by its own mean, so the posterior it draws is not the prior times the likelihood the
  simulation used, and the ranks would test the scaling, not the sampler.
- **Pooling a block's coordinates into one chi-square.** Rejected: the ranks of one simulation's
  coordinates are dependent, and the pooled statistic's null is not chi-square.
- **The log-likelihood as a test quantity** (Modrák et al. 2023). Not taken yet: the coordinates
  cover each parameter, and a test quantity of the data is a later addition to the records.
