# 0030. A lift test's value to the plan, read before it runs

Status: accepted, 2026-10-05; amended 2026-10-05 by ADR 0034: each outer draw's plan and perfect
information's are the search's, the best split on the grid; amended 2026-10-06: the updates' oracle
on a fitted model, CmdStan's posterior refitted at every outer draw (Consequences).

## Context

A lift test narrows a posterior where a panel cannot, and ADR 0015 reads its estimate in the
likelihood. Which test to run, which channel, how far its spend moves, for how long and in how large
a group, is a decision with a price, a heavy-up's spend or the revenue a cut loses, and its return
is the regret it removes from the plan made on what it reads. The engine had no form for a test
not yet run, and no value for one.

Three facts shape the estimator. Most plans of a posterior without a test hold a channel at a
bound, where a local model of the regret, CHC's decision weight
(`chc.allocation.decision_weight`), is zero: a ranking built on it would be blind where the plans
sit, so the value is read by the full preposterior. A test's likelihood is normal, and so bounded:
the Pareto shape PSIS fits to the weights stays low while a handful of draws carry them, and the
effective sample size is the alarm. And the plan a test is valued by is the engine's own, ADR
0019's posterior planner, made again for every outer draw of every candidate.

## Decision

- **`mixdyn.experiments`** holds tests before they run. A `Candidate` is a `LiftEstimate` without
  its lift, checked as one; `Candidate.estimate` is the estimate it reports once read.
  `GroupNoise(market_sd)` sets a group's standard error, `market_sd sqrt(R s / (1 - s))` over a
  readout of `R` periods in a group of share `s`, read against the rest of the market scaled to
  its size. `designs` lays out each channel's candidates: its spend times each multiplier over
  each duration, then a cooldown at the plan, in a group of each share, run after the panel at the
  plan in place (`After`) or inside the history from one of its periods (`Within`), a sum type
  where two optional arguments would allow both or neither. `predicted_lift` is each draw's lift
  at the market's scale as the Bayesian model's likelihood reads the estimate.
- **`value_of_tests(panel, geo, readings, candidates, budget, periods, limits, *, seed,
  cost=Gross(), update=Reweight(), draws=1024, level=0.9, revenue_per_kpi=1.0, uses=1.0,
  grid=512)`** reads each candidate's expected value of sample information (Raiffa and Schlaifer
  1961), `E_θ E_{y|θ}[V(π(p(·|D, y)), θ) - V(a₀, θ)]`, `π` ADR 0019's plan of a posterior's mean
  worths on `grid` spends a channel and `a₀` its plan without the test, by nested Monte Carlo:
  - `draws` outer draws `θ_i` of the readings, each fit's share of them its weight's by largest
    remainders as ADR 0029 shares them, and one standard normal `z_i` each, shared by every
    candidate and drawn from a stream spawned from `seed`, apart from the seed's own. A candidate
    reads `y_i = s μ(θ_i) + se z_i`.
  - Each outer draw's posterior leaves out its own chain of its own fit: `θ_i` drew `y_i`, and
    would weigh most in the plan it is then scored on, while the other chains are independent of
    it. A fit of one chain is refused.
  - **`Reweight(ess=100)`**, the default, weighs each fit's draws by the test's likelihood,
    Pareto-smoothed by `mixdyn.inference.psis`, the lift's relative efficiency over the fit's
    chains standing for the ratios'. The fits' weights are solved again with the test's predictive
    density as one more observation where the readings hold the leave-one-out densities, and kept
    where they were given. An outer draw whose effective sample size, Kish's times that
    efficiency, falls below `ess`, or one of whose weighted fits' shapes passes its threshold, is
    updated as `Gaussian`; each candidate's share of those is its `fallback`.
  - **`Gaussian()`** moves each fit's mean worths by the Bayes-linear update,
    `f̄ + Cov(f, μ) s (y - s μ̄) / (s² Var μ + se²)`, exact where the worths and the lift are jointly
    normal, and weighs the fits by the lift's normal predictive density.
  - The plan after each outer draw's update is scored at `θ_i` exactly, against `a₀` of the
    posterior the update starts from: `θ_i`'s chain left out, the fit's other chains taking its
    weight. Perfect information's plan is `θ_i`'s own worths' plan, so every call returns the EVPI,
    which no test's value passes.
  - **Halves.** `reduction` and its `error` are read on the outer draws at even places, counting
    from 0; `predicted` and the central `level` interval of the realised reduction on those at odd
    places, which no choice by `reduction` read.
  - **The Monte Carlo error** is stratified by fit, each fit's count of outer draws being fixed:
    `sqrt(sum_k n_k s_k² / e_k) / n`, `e_k` the relative efficiency of fit `k`'s values over its
    chains, their split-chain effective sample size over their count, at most 1.
  - **Costs**, `Gross()` or `Net(floor)`, a sum type, from the spend change, `sum(spend -
    planned)`, and the opportunity, `-κ s E[μ]`, the revenue the group is expected to lose, `κ` the
    revenue a unit of the KPI is worth. `Gross` counts each where it is positive, what the test
    puts at risk; `Net` adds them, which a cut of a channel returning less than it costs makes
    negative, and holds the sum at least at `floor`. `per_cost` is `κ reduction / cost`, infinite
    at a cost of 0, and `ranked` orders the values by it; `net_value` is `uses κ reduction - net`.
  - **The local model**, reported and not ranked on: `c' W c / (2 (Var μ + (se / s)²))` at each
    fit's median channels, `W` CHC's decision weight and `c` the channels' parameters' covariance
    with the lift over the fit's draws, summed by the fits' weights. It is NaN on an S-shaped
    curve, which CHC refuses; `model_gap` is its relative miss against `reduction`, logged above
    25 %.
- **`psis` reads its tail in logs.** A precise test read in its predictive's tail weighs draws
  that span hundreds of orders of magnitude, and their tail fits a shape in the hundreds, at which
  `expm1(k t)`, and at times the quantile itself, pass the floats, though their logs, all the
  smoothing needs, do not. A tail whose first quartile lies so near its cutoff that Zhang and
  Stephens' grid passes the floats has no estimate, as one whose first quartile is the cutoff has
  none. Both read as they did, the first's largest ratios clipped at the largest raw one and the
  second's left raw with an infinite shape, without the float warnings that were the only sign of
  either.
- **No effective sample size from split chains of fewer than four draws**, as CmdStan 2.38 reads
  none: it is NaN. Geyer's sequence indexed past its array on halves of two draws, so `diagnose`
  and `loo` failed on four or five draws a chain, the least they accept, and at six or seven read
  a lag that was not there. `loo` reads such draws as independent, as it reads a constant's; so
  does the value's relative efficiency, which the error's strata of a few outer draws reach.
- **The effective sample size reads the draws scaled by a power of two to a range near 1**, which
  changes no bit but the exponents'. The size is free of the draws' scale, their squared
  deviations are not: a lift in units where they underflow, `1e-279` in the example Hypothesis
  found, made the variance 0 for draws that were not one value, and the size 0 / 0.
- **The planner** reads a channel's envelope in one pass where no point lies on or above the chord
  of its neighbours, and orders the stretches by one stable sort. Its plans are the old scan's bit
  for bit, over 24 000 allocations of the tests' mean worths.

## Consequences

- *The oracles.* On the conjugate instance M4 of `voi_conjugate.json`, two channels of a quadratic
  worth over a budget under a normal prior, whose values Maxima derived, the estimator's core with
  the closed-form plan reads EVSI 1/100 and EVPI 1/10 within 4.5 standard errors by either update,
  over 10 000 exact draws in two chains: EVSI 0.00954 and 0.00955, each ± 0.00062, and EVPI
  0.0997 ± 0.0014. The realised reduction's 5 % and 95 % quantiles are Imhof's within binomial
  error (M5), and the reweighted draws' effective sample size averages the share Maxima's law
  gives (M3) within 5 %. On two `tanh` channels whose test is linear in their coefficients
  (`voi_quadrature.json`), `value_of_tests` itself, through the grid planner, `predicted_lift` and
  the exact scoring, reads the quadrature's EVSI and EVPI over CHC's exact plan within 4.5
  standard errors and the grid's bound, by either update: over all 3000 outer draws, EVSI 1.241
  and 1.241, each ± 0.071, against 1.2764, and EVPI 3.570 ± 0.078 against 3.6202. The reweighted
  update falls back on 0.6 % of them. The local model reads 1.281 there, 3 % above the reduction.
- *The oracle on a fitted model* (amended 2026-10-06; `evidence/voi_bruteforce.py`,
  `tests/fixtures/oracles/voi_bruteforce.json`). A tiny world of two `tanh` channels
  over 104 weeks: returns of 2, retentions 0.6 and 0.1, half-saturations 1.0 and 0.7, noise 20.
  CmdStan 2.38's posterior, four chains of 1000, was refitted at each of 200 outer draws for every
  affordable set of its four tests, alone and in pairs: 2000 refits, none with more than 9
  divergent draws of its 4000, every R-hat at most 1.0095. At the same outer draws and normals, each
  set's mean reduction by either update reads its refits' within 1.7 standard errors of their
  paired gap, where the tests hold 4.5: the refits read 0.13 to 5.06, and the largest misses are
  1.69 by `Gaussian` and 1.11 by `Reweight`, both a little high on each set with the fourth test.
  Both updates choose the refits' best set, the second and third tests together. `Reweight`'s
  weights rank each truth uniformly over 28 statistics at the tests' standard errors and at a
  quarter of them (simulation-based calibration), and turn away from uniform where they read the
  standard error doubled or the share a fifth too large.
- *The plan a test is scored against.* Each outer draw was first scored against the plan of every
  draw, which `θ_i`'s own chain moves towards `θ_i` while the update starts without that chain, so
  a test that read nothing lost about the EVPI over a chain's draws.
  - On two tiny worlds by Stan whose plans sit inside their ranges, over 1000 outer draws in four
    chains of 250, a test of a standard error of 10^12 read -0.0046 at an EVPI of 2.4 and -0.011
    at one of 8.7, beside 0.02 to 0.08 for the tests there. On a third, whose plan sat at a corner
    of its ranges, it read 0.
  - Scored against the plan of its own posterior, that test reads 0 at every outer draw; the old
    scoring fails both tests of it.
  - The numbers above are read so. On the quadrature's 1500 draws a chain, EVSI and EVPI each rose
    by 0.006 towards the quadrature's, from 1.236 and 3.564; on M4's 5000 a chain they did not
    move in the digits shown.
- *What the tests hold besides*: no outer draw's reduction passes its perfect information's, on
  random conjugate instances of two and three channels; an outer draw's own chain is left out of its
  posterior, and the outer draw is scored against that posterior's plan without a test, so a test
  of a standard error of 10^12 is worth nothing at every outer draw, on M4's draws as two fits of
  two chains and on two `tanh` channels end to end; a test that reads one fit's lift gives that fit the whole weight when the weights are
  solved again, and given weights stay; on two fits of different counts, spreads and centres, the
  weights solved again are the stacking weights with each fit's predictive density of the reading,
  by either update, computed apart with SciPy's normal; the plan without a test weighs each fit's
  draws by its weight, not its count; an outer draw's effective sample size is Kish's over every
  fit's weighted draws times the lift's relative efficiency; an outer draw whose weights thin below
  100 effective draws is planned as `Gaussian` plans it, and the others are not, and so is one whose
  smoothed tail fits a shape of 0.91 though its weights keep nearly all of 5000 draws; every
  candidate reads the same normals, from a stream spawned from the seed and not the seed's own, one
  seed the same ones again and another seed others; over AR(1) chains of coefficient 0.8 beside a
  fit of independent draws 10 apart, the error is the spread of the mean within 1 %, where the
  draws read as independent would give less than half of it and the gap between the fits read as
  spread more than twice, while antithetic chains are read as independent draws and a fit of one
  outer draw adds nothing; the costs, the halves and the ranking are the arithmetic stated; the
  local model is CHC's decision weight read as stated over two fits, a `tanh`'s scale the one at
  which CHC's curve reads 1/2 at the half-saturation; and the refusals. The planner spends a budget
  at either end of the ranges at that end, and on a tie an earlier channel's stretch first; its
  envelope's vertices are Andrew's monotone chain's, on integer rises whose runs of equal rises put
  points on chords.
- *`psis`*: each of the two tails turns its test red when its fix is reverted, with the float
  warning it removed; the tail's log quantiles are SciPy's generalised Pareto's, at a shape below,
  at and above 0. Every case arviz-stats recorded still holds to `1e-12`. *Short chains*: the tests
  of `diagnose` on four to seven draws a chain and of `loo` on four fail on the old code, at four
  and five by its index and at six and seven by a number, and every case `stansummary` recorded
  still holds. *The scale*: draws scaled by `2^-900` to `2^900` read the unscaled draws' size bit
  for bit, where the old code divided 0 by 0 or overflowed; and the property test of the bound
  passes over 40 seeds of 25 examples each.
- *Mutations*, each run on a copy of the tree from no Hypothesis database at one seed: 44, of
  `value_of_tests` (32), the tail's quantiles (5), the effective sample size (2) and the planner
  (5). 43 failed a test at once. The survivor kept the points on a chord in the one pass; no plan
  told it apart, as the stretches it splits share a slope where the arithmetic is exact, so a test
  of the envelope's vertices against Andrew's monotone chain was added, and it fails that one too.
  - The scoring against each outer draw's own posterior added 6: the plan of every draw, the
    outer draw's chain kept alone, the chain left out of every fit, the fit's other chains not
    given its weight, the fit and the chain swapped, and every outer draw against the first's
    plan. All 6 failed a test.
- No timing is claimed. A candidate's reweighted means are one product a block of outer draws,
  `(outer draws, draws)` by `(draws, channels × grid)`, and a plan an outer draw.
- *Left for later*: `Refit`, refitting a few outer draws' posteriors with the simulated estimate,
  and a multi-fidelity correction of either update, which ADR 0032 builds; the updates' oracle on
  a fitted model, built on 2026-10-06 (above); `choose_tests`, a set of tests under a test budget,
  and the verdict to plan now where no affordable set is worth its cost, which ADR 0031 builds.

## Alternatives considered

- **A reference plan given, `current=`**: dropped. Any other `a₀` adds the same
  `E[V(π(p), θ) - V(a₀, θ)] >= 0` to every test's value, so the ranking does not move, and a check
  of the value's coverage needs the plan without the test. `PosteriorPlan` reports the gains of the
  plan and of the current spend for a reader who wants the difference.
- **The plan of every draw as each outer draw's `a₀`**: rejected. `θ_i`'s own chain moves it
  towards `θ_i`, while the update starts without that chain, so a test that reads nothing would be
  worth less than nothing, by about the EVPI over a chain's draws.
- **A held-out share, `holdout=`**: dropped for halves at alternate places, which spread both
  halves over every fit and chain. An unequal split trades the choice's error for the
  prediction's, and no use asked for that.
- **The Pareto shape as the update's only gate**: rejected, for the bounded ratios above. It gates
  beside the effective sample size, so a heavy tail still falls back.
- **The ratios' own relative efficiency**, an effective sample size for each outer draw and fit, as
  R's `loo` reads `r_eff` from the likelihood's values: the lift's, one a fit, stands for it. The
  ratios are one smooth function of the lift at every outer draw, and reading each would cost an
  autocorrelation for each outer draw and fit.
- **The outer draws read as independent for the error**: rejected. They are a chain's draws in
  order. The oracles' draws are exact, so they could not tell the two apart; a fitted posterior's
  can.
- **The local model as the ranking**: rejected. It is blind on a bound, where most plans sit, and
  refuses S-shaped curves.
- **Cover's iteration to solve two fits' weights again**: ADR 0019's bisection is used, which
  always ends. Three fits or more use Cover's, which can fail to converge, and `value_of_tests`
  raises then.
- **The fits weighed by their models' posterior probabilities**, each times the test's predictive
  density: rejected. The fits are weighed by stacking (ADR 0011), and the engine stacks them again
  after a refit, so the update solves the same problem with one more observation.
- **Regression EVSI** (Strong, Oakley, Brennan and Breeze 2015), a smoother of each outer draw's
  worths on its simulated reading: held back. The smoother's form is one more choice, and its bias
  one more to bound, where the nested estimate needs neither.
- **The exact plan at every outer draw**, by `chc.allocation` or `cvar_allocate`: rejected at a
  solve a draw for every candidate. The grid planner's envelopes plan in one pass, and the
  quadrature oracle bounds what the grid costs.
- **`chc.lift.LiftTest` as a candidate's form**: rejected. The model learns its noise's scale from
  the tests, while a candidate's standard error is set before it runs, by `GroupNoise` or by the
  analyst, as a `LiftEstimate`'s is.
