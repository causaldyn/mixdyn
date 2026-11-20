# 0083. A geo panel planned by the model over geos, and a market plan split by population

Status: proposed, 2026-11-16; adds the arms `dlm-by-population` and `bayes-geo-plug-in` to ADR
0039's harness and declares the rule that chooses the claim's arm on a family whose exports carry a
geo axis; no existing arm, record or output changes

## Context

M2 read the geo families, family 8 (ten geos) and family 10 (three, eight and fifty geos), with
the claim's national procedures: `dlm plug-in` at rung 0 and `dlm two-stage` at the tested rung,
each fitted on the export's market and planning one weekly spend a channel (ADR 0043). The bench
allots such a plan over the geos as the status quo splits each channel (`GeoQuarter.allot`). Its
own equal split splits a channel by population (`GeoQuarter.equal_split`).

M2's records show three things on the two families.

- The market plan lost to PyMC-Marketing's plan of every cell, and to the equal split, at both
  rungs. The status quo's own channel totals, split by population, already remove about 0.037 of
  the status quo's 0.091 regret per euro on family 8 at rung 0: most of the status quo's regret
  there is its geo split, which a market plan keeps.
- The model over geos, `bayes geo plug-in` (ADR 0043), recorded beside the claim, beat every rival
  on family 8 and on family 10 at three and eight geos, at both rungs.
- The model over geos was not run at fifty geos: an earlier costing put one world-rung there at
  about 22 hours on one thread.

So the claim's arm on a geo family is a choice that a run must declare before it runs, between
arms that plan every cell, and a market plan has a better split than the history's.

## Decision

### The rule: a geo family is claimed by the arm that plans every cell, chosen by one rule

- A family whose exports carry a geo axis is read with a claim arm of its own, declared as the
  run's default (`Declaration.default`) on the roots of that family. Family 8, and family 10 at
  three and eight geos, are claimed; family 10 at fifty geos is exploratory, in a root of its own.
- The claim's arm is chosen among at most four candidates, the same at rung 0 and at the tested
  rung, by the selection rule the run's pre-registration states: on the development worlds' fit
  split, for each candidate, family and rival, the mean paired difference of regret per euro,
  candidate less rival; a candidate's score is the largest of these over the families and
  rivals; the smallest score wins; candidates within 0.005 of it tie, and a tie goes to fewer CPU
  seconds a fit, then to fewer settings.
- The candidates: `bayes geo plug-in` and `bayes geo cvar 0.1` (`bayes-geo`, one fit, two
  plans); `dlm plug-in, split by population` (`dlm-by-population`, below); `geo dlm` (`geo-dlm`,
  ADR 0078).
- The development worlds are the previous run's scored worlds: within each family and
  environment its seeds sorted ascending, every world whose rank, from 0, leaves 2 on division by 3
  is held out to calibrate the claim's predictive, and so is every world a capped tool planned;
  the others are the fit split. On M2's records that is 132 worlds of family 8 and 80 of
  family 10 at three and eight geos.
- **On M2's records the rule picks `bayes geo plug-in` at both rungs.** Each score is against
  PyMC-Marketing on family 10, every other comparison lower:

  | candidate | score at rung 0 | score at rung 4 | CPU s a fit, 3 / 8 / 10 geos |
  |---|---|---|---|
  | `bayes geo plug-in` | -0.0015 [-0.0063, +0.0033] | -0.0108 [-0.0197, -0.0018] | 1 725 / 3 095 / 3 660 |
  | `bayes geo cvar 0.1` | +0.0071 | -0.0020 | the same fit |
  | `dlm plug-in, split by population` | +0.0215 | +0.0138 | 462 / 485 / 480 |
  | `geo dlm` (40 of the fit split's worlds) | +0.0306 | +0.0317 | 33 / 92 / 100 |

  The first three candidates' seconds are M2's records on the rented machine; `geo dlm`'s were
  measured on a shared laptop, where the same DLM fit took 1.6 to 1.8 times its recorded seconds.
  On the same 40 worlds `bayes geo plug-in` scores -0.0022 and -0.0120: `geo dlm` is 0.033 and
  0.044 behind it. The second candidate is 0.0086 and 0.0088 behind the first, outside the tie.
  `bayes geo plug-in`'s fit-split mean regret per euro is 0.0135 and 0.0127 on family 8 (rungs 0
  and 4; PyMC-Marketing 0.0293 and 0.0297) and 0.0215 and 0.0181 on family 10 (PyMC-Marketing
  0.0230 and 0.0288). The run's pilot applies the rule once, with `geo dlm` run on the whole fit
  split; until then the pick is the rule's reading of the records that exist, not a selection.

### The market plan split by population (`scorecard.arms.population`, `dlm-by-population`)

- **One variant, `dlm plug-in, split by population`.** The dynamic arm (`scorecard.arms.dlm`) is
  fitted on the geo export's market at each rung, as the `dlm` arm is, and plans its plug-in. Each
  per-geo channel's weekly spend is then split over its geos by population, and a national
  channel's spend is its one cell's. The arm declares `cells = True`, so its record is a plan of
  every cell and the bench scores it as it is.
- **The split keeps each channel's total.** Where a geo's share would put its cell outside its
  box, the channel's cells are moved into their boxes at the channel's total by one shift: the
  nearest point, in spend, of the cells' boxes at that total (`into_box`). On M2's fit split the
  population's split leaves some cell outside its box on 411 of 424 world-rungs.
- CHOICE: the market model reads a plan by its channels' totals and says nothing of where they go.
  A curve of the geos' summed spend is the sum's response exactly where every geo answers through
  one curve of its spend per head and spends the same per head: `x_g = p_g x / P` gives
  `sum_g p_g h(x_g / p_g) = P h(x / P)`, `x` the channel's total and `P` the market's people. That
  is the population's split, at which the model over geos's response of the geos' sum is one channel
  of the sum (ADR 0043). On a concave curve that split also returns the most at the channel's
  total, and it is the bench's equal split's. The history's split, which the bench's allotment
  keeps, is one no model of the sums reads.
- **The claim** is the dynamic arm's plug-in gain, mean and interval: that model reads the plan by
  its channels' totals, which the split keeps. Flags, settings and effective sample size are the
  plug-in's, the settings with `split` and `market plan` added. No forecast and no response: the
  market's are the `dlm` arm's `dlm mixture`'s.
- **What it measures.** On the fit split it lowers the dynamic arm's own regret by 0.030 on
  family 8 and 0.018 on family 10, and beats the equal split on both families at both
  rungs; it still loses to PyMC-Marketing by 0.013 to 0.022. Its fit is the market's, whose cost
  does not grow with the geos: about 480 CPU s at fifty geos as at three.

### The model over geos's plug-in alone (`bayes-geo-plug-in`)

- **`BayesGeoPlugIn`** is `bayes-geo` with one variant, `bayes geo plug-in`: the same fit, the
  same plan, the record kept under the same name, so a run declares one of the two arms and the
  harness refuses both (`scorecard.run.names`). A test holds its record to the full arm's plug-in
  record on one fit, cost aside. It is the one variant two of the harness's arms share; every
  other variant is still one arm's.
- CHOICE: a run that claims the plug-in need not pay for the CVaR plan. On M2's records that plan
  costs 186, 670 and 968 CPU s a world-rung at three, eight and ten geos, 10 to 17 % of the arm's,
  and in an earlier pilot at fifty geos 24 856 on average, more than the fit. Its seconds are the
  search's: on a ten-geo pilot world, over 200 readings, `cvar_allocate` took 1 003 CPU s and
  stopped at its cap of 499 boxes with its gap open, where the plug-in's plan took 32.

### The fifty-geo rung

- **Exploratory.** M2 has no record of the model over geos at fifty geos on its scored worlds.
- **Its cost is the planner's, not the sampler's.** On M2's records the arm's seconds a world-rung
  are the fit's, 1 725, 3 095 and 3 660 CPU s at three, eight and ten geos, then the plug-in's
  plan, 27, 133 and 961 on average. The plug-in plan is `plan_geos`, `chc.allocation.allocate_geos`
  underneath; its seconds have a tail: on family 8 the median world-rung takes 322 and the slowest
  tenth hold 53 % of them, up to 8 527. The gain over the readings, which the plan also computes,
  is about 90 CPU s at ten geos. An earlier pilot of the model over geos, at an earlier commit,
  ran 20 world-rungs at fifty geos: the fit took 8 509 CPU s on average, the plug-in's plan 18 635
  (at most 55 684), the CVaR's 24 856 (at most 44 634). As recorded, the fit costs 1.36 to 1.41
  times as much now at three and eight geos, so about 11 800 at fifty, or 14 700 on a line through
  three, eight and ten geos. A world-rung of the plug-in alone is then about 10 to 13 CPU hours on
  average and, in the slowest world, more than a day on one thread; with the CVaR plan, about 21
  to 24.
- **What would make it cheap is the planner.** The sampler's settings are not changed here: a
  cheaper sampler is a new variant with its own calibration evidence. A profile of `allocate_geos`
  on the slow worlds is the next step.
- **`geo dlm` is not a cheap path there.** On one of the fit split's fifty-geo worlds, on the
  laptop, its fit took 4 323 CPU s and 2.9 GiB at peak, against about 100 CPU s and 0.35 GiB at ten
  geos; its regret per euro there was 0.086, the status quo's 0.069. One world: a cost, not a
  reading.
- **The cheap reading at fifty geos is `dlm plug-in, split by population`**, whose fit is the
  market's, about 480 CPU s at fifty geos. On M2's fit split there, read after the fact and
  exploratory, its regret less the equal split's is +0.012 at rung 0 and +0.000 at rung 4, less
  PyMC-Marketing's +0.007 and -0.005.

## Consequences

- A run's roots of a geo family declare `bayes geo plug-in` as their default at both rungs, and
  hold `bayes-geo` in the root that holds the claimed environments: family 10's three and eight
  geos together, its fifty in a root of their own. The national claim's arms still run there and
  are printed, as every recorded arm is.
- `dlm-by-population` runs end to end under `scorecard/run.py`: a test runs it on family 8's
  first pilot world at rungs 0 and 4 with the dynamic arm stood in, and checks the split, its
  boxes and the record. Refitted on two of M2's development worlds (family 8 at rung 0, family 10
  at eight geos at rung 4), its plan is `split` of M2's recorded `dlm plug-in` plan bit for bit,
  and its claimed gain the recorded one, so its fit-split figures above come from M2's records with
  no refit.
- `geo dlm` runs end to end under `scorecard/run.py` on a family 8 world and on family 10 worlds
  at three and eight geos, at 18 to 150 CPU s a fit on the laptop; its plan is a corner of a
  linear model's (ADR 0078), and it trails the model over geos by 0.03 to 0.04 a euro.

## Alternatives considered

- **The national claim's arms on the geo families.** Rejected: their plan cannot move spend
  between geos. On M2's records the best plan of one spend a channel, allotted as the bench allots
  it, still loses to PyMC-Marketing's plan of the cells at the tested rung, by 0.021 on family 8
  and 0.010 on family 10.
- **The split moved into the box by the bench's one shift over every cell**, as the bench moves a
  plan off its box. Rejected: that shift moves spend between channels, which the market model
  chose. On the fit split it scores 0.006 to 0.010 lower than the split that keeps each channel's
  total, read after the arm's definition was fixed; it would not change the rule's pick.
- **A name of its own for the plug-in planned alone.** Rejected: the rule reads the previous run's
  `bayes geo plug-in` records, and the same plan under a second name would be a second candidate
  in every comparison that joins the two runs.
- **Every variant of the dynamic arm split by population.** Rejected: one candidate a fit, the
  plug-in, as the evidence measured; the other variants are recorded by the `dlm` arm.
- **The fifty geos in the claim.** Rejected for now: no record of the model over geos exists on
  the scored worlds there, and its cost (above) puts its slowest world-rung past a day on one
  thread, longer than a spot instance is safely held.
