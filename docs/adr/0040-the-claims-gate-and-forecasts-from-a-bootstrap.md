# 0040. The claim's gate, the worlds a family needs, and forecasts from a bootstrap

Status: accepted, 2026-10-11; amended 2026-11-07 by ADR 0079: a model's forecast of periods held
out of its fit holds out the window's last periods by default, not periods drawn from all of it;
amended 2026-11-10 by ADR 0080: `claim read` is handed every root of a run, a pool's runs may test
at different rungs above 0, the tools of the roots beside a run are read with it, and
`claim read --recorded` prints every recorded arm less every rival after the claim; amended
2026-11-16 by ADR 0081: a run may declare its claim read by family and rung, the verdicts
tested in a declared fixed sequence, its comparisons predicted to fail read and left out

## Context

The scorecard's claim is that the engine's default procedure plans no worse than every other arm,
on every family of worlds, by more than a margin. Its runs so far printed only unadjusted
verdicts: each arm less each other arm, read by where the paired 95 % interval lies against 0.

The bench already holds the gate such a claim needs (`causaldyn_bench.scorecard.gate`): `Gate`
reads a paired difference against a margin at looks fixed before the run, `crossings` gives its
chances at a known drift, `level` gives a comparison's sequential level, and `holm` steps down over
those levels. `Gate` arrived at the bench's `dfbdf17`, `level` and `holm` at `aca4eda`. The drift
run's bench, `1f4161c`, holds all of them, and they are unchanged up to `d1c421a`.

The drift run's reading showed what its forecasts lacked: 76 of its 109 forecasts held, against
87.4 expected. Three things made the forecasts, or the reading of their count, surer than they
should have been:

- Ten pilot worlds a law, and a predictive that took the pilot's differences as normal, where the
  worlds are heavy-tailed.
- Forecasts on items whose arm the pilot had chosen: the engine's default procedure, by the rule
  that kept it, and the PyMC-Marketing setting of least pilot regret. The pilot that chose an arm
  for its showing also forecast that showing.
- The count's spread taken as a sum of independent items, where most items read the same worlds
  and the same arm.

## Decision

### The claim (`scorecard.claim`)

- **A run declares the claim it is read by**: the engine's default procedure (`default`, one of
  its arms' variants) and the margin `delta` in regret per euro (`margin`), together or neither.
  `declare` refuses a default that is not one of the run's own variants.
- **The rivals** are every arm whose records are not this engine's: the built-in arms and the
  run's tools. The engine's other variants are its own alternatives to the default, chosen between
  by the pilot. The report's unadjusted tables still read them.
- **A comparison** is the default's regret per euro less a rival's, paired over every environment's
  declared worlds at one rung, read by the bench's `Gate` at one look. It is within the margin
  where the paired 95 % Student-t interval's upper end lies below `delta`: at one look the gate's
  boundary is Student's t at 0.975, so this is the same reading. A run that does not hold its
  declared worlds whole in every environment leaves its comparisons unread.
- **The claim is an intersection-union test** (Berger 1982). It holds only where every comparison
  of every run, at every rung, is within the margin, and an unread comparison is not. Each
  comparison is read at one-sided 0.025, and so is the claim, with no adjustment.
- **The losses.** Each comparison is also read the other way, the rival less the default against
  0. Its verdict is printed unadjusted. Beside the verdicts, the losses that Holm's step-down shows
  at familywise 0.025 over the claim's comparisons are listed, each at its `Gate.level`: at one
  look, the Student-t p-value of the rival beating the default. An unread comparison holds its
  place in the step-down at level 1.
- **An exploratory run** is read and printed, and is left out of the claim and of Holm's list.
- **A tool's cap** (amended). A run may declare, for any of its tools, a cap on the worlds an
  environment that tool is read on (`caps`, at least 2). A capped comparison is read over the
  first `cap` declared worlds of each environment, the rest of the run over all of them, and a
  run that does not hold that prefix whole leaves it unread. `exploratory_caps` leaves the capped
  comparisons out of the claim and of Holm's list, as an exploratory run is left out. The joint
  sizing reads a pilot's declared caps as it reads `--cap`; a tool's `--cap` replaces its declared
  one. Both fields are additive: a declaration that names neither is written as it was before
  them, so an older run reads and is declared again byte for byte. The runner still plans every
  world of a capped tool; the cap is how the claim reads it.
- **Where it is printed.** A run's report ends with the claim read on its own comparisons.
  `python -m scorecard.claim read RUN ...` reads it over many runs, which must declare one margin.

### The worlds a family needs (`claim size`)

- At a true tie, a comparison of paired SD `sd` over `N` worlds is within `delta` with chance 0.8
  where `N = ((z_0.975 + z_0.8) sd / delta)^2`. Maxima gives `z_0.975 + z_0.8 =
  2.80158521811296844...` from `erf` at 40 digits; the code takes the sum from SciPy, and a test
  holds it to Maxima's to `1e-12`.
- `size` reads a pilot run's scores. For each rung and rival it takes the paired SD of the default
  less the rival over the pilot's declared worlds. The largest binds. `N` is the fewest worlds in
  tens that reach 0.8 at that SD, and at least 10.
- `N` counts every environment's worlds together, as the gate pools them. A run declares
  `ceil(N / E)` worlds in each of its `E` environments.
- **The cap is 520.** It is the earlier design's figure: 80 % at a true tie at paired SD 0.081 and
  margin 0.01 is `(2.8016 * 0.081 / 0.01)^2 = 514.96` worlds, 520 in tens. A family whose `N`
  passes it runs at the cap, is declared `exploratory`, and is left out of the claim.
- `size` prints each family's `N`, the worlds an environment, the binding rival, rung and SD, and
  the power at a true tie on those worlds as the bench's `Gate.crossings` computes it. A test holds
  that power to `Phi(delta sqrt(N) / sd - z_0.975)` and checks that 10 fewer worlds fall below 0.8.

### Forecasts from a bootstrap (`scorecard.forecast`)

- **Items.** A run's own module names its items: a verdict, a clause or a share over some
  environments at one rung (`Item`), as the drift run's were. `pilot` lays out the pilot's values
  one row a world, with every rung of that world on the row. `quantity` reads the axes that every
  record scores. A run whose items read more passes its own reader.
- **Two stages, one stream.** Each resample draws, in each environment, the pilot's worlds with
  replacement, as many as the pilot holds. From those it draws as many worlds as the scored run
  reads there. The first stage is the pilot's uncertainty and the second is the scored run's own
  draw, so the scored mean spreads about `s^2 (1/n + 1/N)`, as the drift run's predictive does.
- **Paired, stratified, shared.** Worlds are drawn whole, every arm and rung of a world together.
  Each environment is drawn on its own, as the scored run reads a fixed count of each. Every item
  reads the same resample.
- **The forecast.** On each resample an item is read as the scored run will read it. A verdict's
  forecast is the verdict most resamples read, and its chance is that share. A share's forecast is
  the central 90 % of the resampled counts, and its chance is the share of resamples inside it.
- **No forecast** is made on an item that reads an arm the pilot selected, named by the caller
  (`selected`). None is made on an item whose pilot values are all one value either. Every resample
  would read such an item alike, so its chance would be 1 by construction, not by evidence.
- **The count of held forecasts.** On each resample, count the forecasts the resample holds. The
  mean of these counts is the expected count, the sum of the chances. Their central 95 % is the
  count's interval, and it carries the items' correlation.
- **At least 2 000 resamples**, 4 000 by default, from a seed the caller declares.
- **`scorecard/drift.py` does not change.** The drift run's pre-registration reads it as it was.

### Amended 2026-11-07: a holdout is blocked

The forecasts above are a bootstrap of a run's items, and hold out no periods of any fit. A
model's own forecast check does: `mixdyn.models.bayes_mmm.holdout_bayes` fits the panel with some
of its window's periods left out and scores the posterior predictive there by its CRPS (ADR 0079).
**It holds out the window's last periods by default (`kind="blocked"`).** Periods drawn from the
whole window (`kind="interleaved"`) each sit between fitted neighbours, and where the noise is
autocorrelated, a term that bends with time, a trend of many knots or a season, fits the noise
they share. The score then reads a forecast no planner gets, one made from inside the window. On
four worlds of 104 weeks, 13 held out, a trend of 12 knots: with AR(1) noise at 0.95 the
interleaved holdout's CRPS was 0.07 to 0.28 of the blocked one's, and 0.19 to 0.42 of its own on
the same worlds with independent noise of the same SD. With the 3 knots the scorecard's Bayesian
arm uses, the blocked CRPS was 1.10 and 1.34 times the interleaved one on two of those worlds: the
leak needs a term that bends enough to follow the noise. The interleaved kind stays, to show the
leak, and is not a default anywhere.

## Consequences

- **The drift run's pilot, forecast again by the bootstrap** (seed 20261011, 4 000 resamples,
  100 scored worlds a law), and read against the scored run's reading:
  - 13 of the 109 items carry no forecast, because their pilot values are one value: the seven
    failure shares, the past-the-bounds shares of `dlm mixture` and PyMC-Marketing's two
    time-varying settings, Meridian's moves at both rungs, and the DLM's coverage on `season`.
    The old rule forecast all 13, and all 13 held.
  - On the other 96 items the bootstrap expects 77.5 to hold, 95 % interval [69, 85]. **60
    held.** The old rule expected 75.05 on those 96, and 63 held.
  - The count's SD is 4.44, against 3.19 if the items were independent: the shared worlds widen
    its interval by about 1.39 times.
  - Leaving out every item that reads an arm the pilot selected (`dlm mixture`,
    `pymc-marketing shared`) leaves 17 forecasts. The other 92 items are 84 selected and 8 with one
    value. The 17 expect 16.1, interval [14, 17], and **17 held**.
  - A second seed (7) gives 77.4 [69, 85] and 61 held on the 96, and 16.0 [14, 17] and 17 held on
    the 17.
- **So the bootstrap alone does not mend a pilot of ten worlds a law.** The 60 held lie below its
  interval. The misses are in the items that read the selected default. Leaving those items out
  leaves only the plain items: lengthscales, coverage shares and claimed gains far from 0. They
  held as forecast, but they test the rule little.
- **A bootstrap is narrower than Student's t on a small pilot.** A resample's mean spreads as
  `(n - 1)/n` of the pilot's variance. The pilot part of the Student-t predictive spreads as
  `(n - 1)/(n - 3)` of it. The ratio of their SDs is `sqrt((n - 3)/n)`: 0.84 at 10 worlds and
  0.95 at 30, by Maxima. The rule is meant for pilots of at least 30 worlds an environment. It does
  not enforce that.
- A share that the pilot never saw vary gets no forecast. A pre-registration that wants one, as for
  failures, states it as a bound outside the count of forecasts.
- A family whose pilot SD is large becomes exploratory, and the claim is stated on the other
  families only. At margin 0.05 the cap binds above a paired SD of `0.05 sqrt(520) / 2.8016 =
  0.407`, by Maxima.
- The declaration gains three fields. A declaration written before them still reads, with no claim.
  `compare` refuses a run that declares none.
- Two runs of one family, such as a tool's later run on some of its worlds, are two runs to `claim
  read`. Their comparisons count apart in Holm's list.

## Alternatives considered

- **One stage, the pilot's resample read as if it held the scored run's worlds.** This omits the
  scored run's own draw. At `n = N` the predictive's variance would be halved. A test holds the
  two-stage chances to the drift rule's at 300 normal pilot worlds and 300 scored, and the one-stage
  draw fails it.
- **Draw the scored run's worlds from the pilot directly.** This omits the pilot's uncertainty, and
  its chances would tend to 1 as `N` grows.
- **Keep the beta-binomial with Jeffreys' prior for shares.** It forecasts a share the pilot never
  saw vary, but the forecast then comes from the prior, not the pilot. Each item's beta draw is also
  its own, so the share items would not share the count's resamples. Rejected; such a share carries
  no forecast.
- **Student-t widening of the bootstrap**: scale each resample's deviation from the pilot's mean
  by `sqrt(n / (n - 3))`. That restores the t spread on small pilots, but it adds a parametric
  assumption to a rule chosen to have none. A pilot of 30 worlds an environment makes the
  difference 5 % in SD. Not taken.
- **Rivals that include the engine's other variants.** The claim is about other systems' arms.
  The default's siblings are the choice the pilot made between them. Gating against them would
  test that choice again, and a sibling that plans far from the default would bind every family's
  `N`.
- **Sequential looks.** The bench's gate reads several looks by Lan and DeMets' spending. Not
  taken: the run's `N` is fixed from the pilot at one look.
- **Holm over each run alone.** The list's family is the claim's comparisons, so `claim read` steps
  down over every run it reads. A run's report labels its own section as that run's alone.
