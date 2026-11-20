# 0031. A set of lift tests chosen by its value, or none

Status: accepted, 2026-10-05

## Context

ADR 0030 values one test at a time. A market can hold several at once, in disjoint groups: a
go-dark of one channel beside a heavy-up of another. Two tests together are not worth the sum of
their values. Two that read one quantity overlap. Two that each read a mix of two channels can tell
apart what neither tells alone. Which set to run is bounded by a test budget and by the market,
whose untested rest is the control the tests are read against. And running none is always a
choice: a test worth less than it costs is a loss, however much regret it removes.

## Decision

- **Tests read together** are independent given the draw, so their likelihoods multiply:
  `y_t = s_t μ_t(θ) + se_t z_t`.
  - `Reweight` weighs each draw by the product and smooths within each fit as before. It solves
    the fits' weights again with the tests' joint predictive density, the mean of the product over
    the fit's draws. A fit's relative efficiency is the least of its tests' lifts'.
  - `Gaussian` is the vector Bayes-linear update, `f̄ + Cov(f, μ) S (S Var μ S + D)⁻¹ (y - S μ̄)`,
    `S` the shares and `D` the noise variances. It weighs the fits by the normal density of `y`.
  - The local model reads `tr(W c (Var μ + D / S²)⁻¹ c') / 2`, `c` now the parameters' covariance
    with each test's lift.
- **`ExperimentValue.candidates`** is the tuple of tests read together; `value_of_tests` reads one.
  `predicted_net_value` is `uses κ predicted - net`, the net value on the outer draws at odd
  places.
- **`choose_tests(panel, geo, readings, candidates, budget, periods, limits, *, test_budget, seed,
  cost=Gross(), update=Reweight(), draws=1024, exhaustive=256, level=0.9, revenue_per_kpi=1.0,
  uses=1.0, grid=512)`** returns an `ExperimentChoice`:
  - **Affordable sets.** A set is affordable where its shares sum to at most 1/2 and its cost,
    under `cost`, is at most `test_budget`. Each sum is its terms' exact sum rounded once
    (`math.fsum`), so no order of the tests moves a set across a limit: summed in turn, shares of
    0.05, 0.17 and 0.28 come to 0.5 in one order and past it in another.
  - **Normals.** Each candidate reads its own column of normals, the same in every set it is in,
    from the stream spawned from the seed. Two sets differ by their tests, not by their draws of
    noise. Two tests of one set read independent noise, as their likelihood says they do; one
    column for every candidate, as `value_of_tests` reads them, would have them read the same.
  - **Search.** Where at most `exhaustive` sets are affordable, every one is valued. A depth-first
    search finds them, trying the candidates by their net costs, the least first, and growing only
    an affordable set. In that order a set's share and gross cost only grow, and its net cost falls
    only while all its tests' are negative, where `Net` charges its floor: no set grown from one
    that costs too much is affordable.
  - Otherwise the set grows greedily from none. Each step adds the affordable test whose value adds
    the most a unit of the cost it adds, the larger addition on a tie. It stops when no addition is
    worth more than the net cost it adds. The grown set is kept where it is worth more than the best
    affordable single test, else that test is.
  - `best` is the affordable set of the largest net value, read on the outer draws at even places,
    and `single` the best single test. `values` holds every set valued, in the order valued, so
    the sets behind the best can be read beside it.
  - **The verdict** is to test where `best.predicted_net_value > 0`, read on the outer draws at odd
    places, which no choice read. It is to plan now otherwise, and where no set is affordable. The
    largest of up to `exhaustive` noisy net values overstates the set it picks; the other half does
    not.

## Consequences

- *The oracle.* On M4's instance, the sum and the difference of the two channels' parameters are
  each read with noise of variance 1/2 (`voi_conjugate.json`, M6). Maxima derives their values, 1/100
  and 1/15 apart and 19/280 together, and the second's after the first, 81/1400, by the chain rule.
  The test suite recomputes all four in exact rationals by hand, with the pair's posterior covariance.
  - The estimator's core reads them within 4.5 standard errors by either update. Over 10 000 exact
    draws in two chains it reads 0.00928 and 0.00925 ± 0.00062, 0.06599 and 0.06596 ± 0.00133, and
    0.06717 and 0.06715 ± 0.00134.
  - The pair's reduction less the first's at each outer draw reads 0.05789 and 0.05790 ± 0.00120.
  - The reweighted update falls back on 0.7, 0.3 and 2.8 % of the outer draws.
- *The joint update.* Two tests on two fits, the second's lift partly the first's: each outer
  draw's weights are the stacking weights with both readings as one more observation. The densities
  are computed apart with SciPy's normal and multivariate normal, and match to `1e-9`.
  - The local model of the pair is the trace above computed apart, and above the single test's.
  - Beside a test of independent lifts, the effective sample size reads the least efficiency.
- *The choice's tests.*
  - The affordable sets are every set within half the market and the test budget, by brute force
    over 300 Hypothesis examples of up to seven tests, under both costs and with tests of negative
    net cost.
  - The greedy set grows as stated in eight traces of the sets it values. One trace is a pair worth
    only together, which it never reaches.
  - End to end, three tests on the cheap world make four affordable sets under a test budget of 60.
    Every one is valued, `values` holds them in the order valued, the best is the largest net value
    among them, and each test reads its own column of normals in every set.
  - Grown greedily, the chosen set is worth no more than the best of all, and `values` holds each
    step's sets in turn.
- *A test worth less than its cost is not run.* On that world the tests' net costs run from 2.0 to
  14.3, and every set costs more, net, than the regret it removes is worth. The best set is the
  heavy-up, which loses least: it
  removes some regret, but less than it costs, and the verdict is to plan now. A thousand uses of
  what the tests read would pay for them, and the verdict is then to test. With no test affordable
  it is to plan now.
- *Mutations*, each run on a copy of the tree from no Hypothesis database at one seed: 29, of the
  joint update and the local model (6), the normals (2), the search and its sums (11), the greedy
  growth (5), and the choice and its verdict (5). 28 failed a test.
  - The survivor summed the shares by Python's `sum`. It has compensated its rounding since 3.12,
    and reads every order of up to five shares of whole hundredths that make 1/2 as 1/2; summed in
    turn, the shares fail.
  - A first search let a set's later tests of negative net cost bring its cost back down, and the
    mutant that dropped that survived: by net cost, the least first, no such test ever follows a
    set it could bring back. It was removed. Each of three other orders the search could try fails
    the test of the affordable sets.
- Some arithmetic is not tested, because it cannot move a value. The constants of the densities,
  `k log(2π) / 2` and the standard errors' logs, are the same under every fit, and a stacking weight
  is unmoved by a constant added to one observation's densities. That the search prunes moves only
  its speed: grown on from the sets that cost too much, it would find the same sets.
- No timing is claimed. A set's update costs what one test's did, its likelihood a product of `k`
  normals; the exhaustive search values at most `exhaustive` sets, the greedy one at most the
  candidates' count a step.
- *Left for later*: sets of tests in several geos, which wait for estimates over geos (ADR 0015).
  `Refit` and the multi-fidelity correction, as ADR 0030 left them, are ADR 0032's.

## Alternatives considered

- **A set's value as the sum of its tests'**: rejected. It counts what two tests share twice, and
  misses what only the pair tells.
- **The verdict on the choice's half**: rejected. The best of many noisy net values is biased
  upward, so a set worth nothing would be run on its noise.
- **A margin on the verdict**, testing only where the predicted net value is above its error:
  not taken. The verdict is the decision that maximises the expected value under the posterior, and
  the reduction's error is reported beside it.
- **Exhaustive search only**: rejected at a number of sets exponential in the candidates. **Greedy
  search only**: rejected, as it misses a pair worth only together. The switch is at 256 sets, so
  up to eight candidates are always searched in full.
- **The share of the market a set may take as a parameter**: not added. Half keeps a control as
  large as the tests, and no use has asked for another.
- **One column of normals for every candidate in a set**, as `value_of_tests` shares them: rejected.
  Two tests in one set would read the same noise, which their joint likelihood takes as independent.
