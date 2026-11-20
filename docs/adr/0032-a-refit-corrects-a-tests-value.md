# 0032. A refit corrects a test's value

Status: accepted, 2026-10-05

## Context

ADR 0030 values a test without refitting the model: `Reweight` weighs the posterior's draws by the
test's likelihood, and `Gaussian` moves their mean worths by the Bayes-linear update. Each reads
the posterior the plan would be made on once the test has run only as far as the draws cover the
reading and the model reads the test as its likelihood does. The posterior the plan is made on is
the model refitted with the test's estimate. A refit costs a fit, so one at every outer draw of
every set is out of reach, and a few are not. A few refits can correct the mean of many cheap
updates: the mean of the cheap reductions plus the mean gap between refitted and cheap reductions
where both are read, a multi-fidelity estimate (Peherstorfer, Willcox and Gunzburger 2016). Heath,
Manolopoulou and Baio (2018) likewise spend a few nested fits on a cheap EVSI, matching its
moments rather than correcting its mean. ADR 0030 left it for later.

## Decision

- **`Refit(draws, fit, then=Reweight())`** is an update.
  - Every outer draw is valued by `then`, as before.
  - At `draws` outer draws of each half, evenly spaced from its first to its last, the readings
    the update read become the candidates' estimates (`Candidate.estimate`), each channel's in the
    candidates' order. `fit` takes them and returns the readings of the model refitted with them,
    read beside the estimates it already reads. Their plan, made as the plan without a test is, is
    scored at the outer draw's truth.
  - Each half's mean is corrected by its refits' mean gap, `Ū + mean_M (R - U)`, `U` the update's
    reductions, `R` the refits' and `M` the outer draws refitted. Its standard error is
    `sqrt(e² + Var(R - U) / m + 2 Cov(U, R - U) / n)`: `e` the error of `Ū` as before, `m` the
    refits and `n` the outer draws of the half. The covariance enters because the refits are
    outer draws of the half, not draws apart from them.
  - The interval of the odd half moves by its correction.
  - Each set's refits are logged at INFO: their count, the reduction's correction beside its error,
    and the prediction's. A correction of several errors says the cheap update misreads the model.
- **`ExperimentValue.refitted`** is the refits' reduction at the outer draws refitted and NaN at
  the others; NaN throughout under the other updates. Under `Refit`, `reductions` and the update's
  diagnostics are `then`'s.
- A refit is a fresh fit, independent of the outer draw's truth given the data, so it leaves no
  chain out.
- Refused: `Refit.draws` not a whole number, at least 2, as a variance needs two; more than the
  outer draws of a half; and a refit whose readings have other channels, or the same in another
  order, as its plan would be scored against the wrong channels.

## Consequences

- *The oracle.* M4's test refitted exactly, the conjugate posterior mean planned, at 200 outer
  draws of each half of 10 000:
  - each refit reads the reading the update read there;
  - the update, all but exact on this instance, misses the refit with a standard deviation 0.021
    of the refitted reductions' under `Gaussian` and 0.064 under `Reweight`;
  - the corrected value reads EVSI 1/100 within 4.5 standard errors under either: 0.00949 and
    0.00960 ± 0.00065 and 0.00068, against the update's 0.00954 and 0.00955 alone.
- *The error.* Cheap reductions 0.5 above the refits', with noise of their spread, over 2000
  replications of 400 outer draws, every fourth refitted: the corrected mean's miss over its error
  has mean -0.013 and variance 0.945, and 90.6 % of the misses are within 1.645 errors. Without the
  covariance the variance is 0.64. The variance's degrees of freedom move it by 0.01, and are not
  tested.
- *The wiring*, by a fit that records what it is given, under either `then`:
  - it is called at the places stated, for each candidate or set in turn, with each test's
    estimate of the reading the update read, by channel in the candidates' order, each test of a
    set reading its own normals;
  - a fit that returns the readings it was given plans as without a test, so its reductions are
    0, and each half's mean, its error and the interval move as stated;
  - `then`'s reductions and diagnostics are kept;
  - a `Refit` may take every outer draw of a half, and its mean is then the refits' alone.
- *Mutations*, each run on a copy of the tree from no Hypothesis database at one seed: 27, of the
  refusals (4), the estimates a refit reads and at which draws (8), the plan it makes (2), how a
  value reads its refits (7), and the corrected mean and its error (6). All 27 failed a test.
  - Two survived a first pass: a `Refit` that valued its outer draws by `Reweight` whatever its
    `then`, and the halves' strata swapped in their errors. The wiring is now tested under both
    updates, and the halves with their outer draws in different fits.
- *What it costs*: `2 draws` fits for each set valued. `choose_tests` values up to `exhaustive`
  sets, or the candidates' count a step of its greedy search, so a caller bounds the fits by
  `draws` and `exhaustive`. No timing is claimed.
- *Left for later*: refits spent where they can change the choice, on the sets whose cheap net
  value is within a few errors of the best's; places chosen by the cheap reductions' spread; and
  the updates' oracle on a fitted model, built on 2026-10-06 (ADR 0030, Consequences).

## Alternatives considered

- **A refit at every outer draw**: rejected, `2 n` fits for each set.
- **The refits alone**: rejected. Their mean's error is the refitted reductions' spread over
  `sqrt(m)`. The corrected mean's is the gap's over `sqrt(m)` and the update's over `sqrt(n)`,
  which a close update makes much smaller.
- **A fitted coefficient on the gap**, the control variate's optimum: not taken. It needs the
  covariance of `U` and `R` from the same `m` refits, which biases the mean at small `m`; a gap
  of coefficient 1 is unbiased at any `m`.
- **A ratio correction**, `Ū` times the mean of `R / U`: rejected, as a reduction can be 0.
- **The correction on the choice's half alone**: rejected. The prediction would be the update's,
  biased as the correction exists to undo.
- **Places at random**: not taken. Evenly spaced places give each fit and chain its share of the
  refits, as stratifying would; at random, the shares would vary from call to call.
