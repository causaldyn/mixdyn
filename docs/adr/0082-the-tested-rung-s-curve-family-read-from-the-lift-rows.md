# 0082. The tested rung's curve family read from the lift rows

Status: proposed, 2026-11-16. The number may move when the M3 branches are integrated. Adds
three opt-in settings, each off by default: `mixdyn.decisions.pick_by_lift_tests` (named
`lift_row_pick` until 0.3.0), a Hill's slope prior on `HillCurve`, and a kernel's length on
`BayesSpec`; and the scorecard's arms that read them, `bayes-concave` and `bayes-lift-pick`, and
`dlm-two-stage-lift-pick`. No default changes.

## Context

At M2's tested rung the claim's arm, `dlm two-stage`, lost to PyMC-Marketing on every national
confirmatory family. Read after the run, on M2's own records:

- **The loss is a tail, and the tail is the Hill.** The two-stage's first stage is the static
  model's two fits, a Hill and a `tanh`, and it takes the family of the larger stacking weight.
  In worlds whose curves are concave the Hill took the weight in 39 % of worlds, and those worlds
  carried 88 % of the summed gap to PyMC-Marketing (a mean of +0.081 a euro against +0.007 where
  the `tanh` took it).
- **The weight rests on noise.** Stacking reads every observation, the 156 weeks of the KPI and
  the 12 lift rows. The two families' leave-one-out densities differed by less than 2 nats in
  74 % of those worlds, each fit's own standard error about 10, and the weight still jumped past
  0.8 as soon as the difference changed sign.
- **The mechanism.** A go-dark test reads a channel's chord, its return from no spend to the
  spend tested, and every tool read the chord to within 3 %. The Hill's slope turns that chord
  into the margin the plan reads: its posterior median slope on paid search was 1.75, a curve
  whose margin exceeds its chord, so the margin came out 31 % too high and the plan filled that
  channel.
- **Planning the same fits from the `tanh`** cut the regret of the Hill-weighted worlds by 0.093 a
  euro [-0.150, -0.036] (39 worlds, the frozen code reproduced to the bit). It is not a rule: where
  the truth is S-shaped it added 0.139 (4 worlds). Picking the family by the lift rows alone
  recovered about 58 % of that change and kept the Hill in all 4 S-shaped worlds.
- **The tests were read twice.** The two-stage reads them in the static fits and again in the DLM.
  It planned within 0.005 a euro of `stacked`, the static arm's plan on the same fits, on every
  family, and its returns' 90 % intervals covered 0.70 to 0.78 of the truths against `stacked`'s
  0.84 to 0.89.
- **The static kernel runs over the panel**, 156 weeks, where the worlds' generator and
  PyMC-Marketing's `l_max` stop at 6. At the decays the fits settle on, 0.3 to 0.6, the weights
  past lag 6 hold 0.07 % to 4.7 % of the kernel, so the length was expected to matter little; it
  was never measured.

## Decision

- **`lift_row_pick(fits)`** (`mixdyn.decisions`): the whole weight on the fit whose PSIS-LOO
  density, summed over the observations past the KPI's (the geo tests' gaps, the lift estimates,
  the lift series), is the largest, the first on a tie; None where the fits read none of them.
  `stacked(fits, weights=...)` reads the weights as any weights. It is a hard pick, not stacking
  over the lift rows: stacking over 12 rows reads the mixture the planner would then plan for,
  and a mixture of the two families' plans recovered none of the change in M2's refits.
- **A Hill's slope prior is its curve's**: `HillCurve(slope_location, slope_scale)`, the slope
  `LogNormal(slope_location, slope_scale)`, by default `(0, 0.5)`, as before. The static model,
  the geo model, the health card's prior draws and the factsheet's priors read it. The candidate
  `CONCAVE`, `LogNormal(-0.3, 0.4)`, has its median at 0.74 and 23 % of its mass above 1, 3.9 %
  above 1.5 and 0.7 % above 2, against 50 %, 21 % and 8.3 % for the default.
- **A kernel's length is the spec's**: `BayesSpec.kernel`, by default None, the longest series the
  model reads, as before. Stated, every row of lags keeps its first `kernel` periods, lag 0 first,
  and the weights are normalised over them; a kernel longer than every series runs on with no
  spend before the panel. The nested model refuses it: its stages' kernels are as long as the
  longest series either reads.
- **The tested rung's candidates** (`scorecard.arms.tested`): `stacked`'s plan, the static arm's,
  each candidate stating its rule (stacking or the lift rows' pick), its Hill's slope prior and
  its kernel. The tests are read once, in the static fits, and the DLM's second reading is gone.
  Each candidate is held by one arm, as each variant's records are its own: `bayes-concave`
  holds both concave candidates, stacking's and the lift rows' pick, which read the same two
  fits, so the pick costs a plan and not a fit; `bayes-lift-pick` holds the pick over the default
  prior. An arm draws each fit its candidates read once, from the seed the `bayes` arm draws its
  own from, so a candidate's plan does not depend on the arm that holds it. A record names its
  candidate, the rule it read, and each fit it read with that fit's CPU seconds.
- **The two-stage arm's rule** (`DlmTwoStage.rule`): `"stacking"` by default, as before;
  `dlm-two-stage-lift-pick` picks its first stage's family by the lift rows. It is there for a
  pilot that wants it; the candidates above do not read the DLM.

## Screening

Before the box pilot the candidates were screened on 38 worlds of M2's scored worlds held for
tuning: families 0, 2, 3, 5 and 7 at the tested rung (family 7 at k = 2) and the S-shaped families
4 and 14, the question and the rule written before the run. Refitted, `stacked` reproduced M2's
recorded regret exactly in all 38. Each candidate's regret less `stacked`'s, a euro, mean and 95 %
interval (the concave worlds pooled over the Hill-leaning and `tanh`-leaning worlds by their share
of the worlds held for tuning):

| Candidate | Concave, pooled | Hill-leaning (20) | S-shaped (10) |
|---|---|---|---|
| `stacked, lift-row pick` | -0.023 [-0.054, +0.008] | -0.060 [-0.147, +0.027] | +0.030 [-0.115, +0.175] |
| `stacked, concave hill` | -0.017 [-0.035, +0.002] | -0.043 [-0.096, +0.009] | +0.015 [-0.128, +0.157] |
| `stacked, concave hill, lift-row pick` | -0.027 [-0.058, +0.004] | -0.070 [-0.157, +0.017] | +0.030 [-0.115, +0.175] |

No candidate changed a plan in the 8 `tanh`-leaning worlds. The rule kept a candidate whose pooled
concave change is below 0 and whose S-shaped change is below +0.02: `stacked, concave hill` alone.
The lift rows' pick took the S-shaped worlds' weight off the Hill in 4 of 10 (stacking's Hill
weight was at least 0.5 in 8, the pick's in 4) and paid +0.030 there. On the claim's families it
did best: its largest family mean less PyMC-Marketing's was 0.070 (family 7, 4 worlds), against
0.197 for `stacked, concave hill` and 0.235 for `stacked`. The S-shaped families are exploratory,
so a pilot that scores the claim's families alone does not read that cost. The pilot carries
`stacked, concave hill, lift-row pick` beside `stacked, concave hill`, both from one draw of
`bayes-concave`'s fits, and reads the same S-shaped guard on families 4 and 14.

A 6-week kernel was screened on 16 of those worlds. Its fits took 96 and 46 CPU seconds a fit on
average (the Hill, the `tanh`) against the panel kernel's 235 and 153, but in two concave worlds it
raised `stacked`'s regret by 0.22 and 0.40, in one by moving the stacking weight from the `tanh` to
the Hill; with the lift rows' pick it added +0.044 [-0.040, +0.128] in the 12 concave worlds
against the panel kernel's, past the rule's +0.005. The kernel stays the panel's.

## Consequences

- No default moves: `bayes`, `dlm-two-stage` and every model fitted without the new settings plan
  and record as before, M2's records among them.
- At rung 0 there is no lift row, so a candidate that picks by them stacks; the record says so
  (`rule read`).
- A Hill whose prior leans concave cannot follow an S-shaped truth as readily; a pick by the lift
  rows can still keep the Hill where the tests read an S-shape. Which candidate carries the tested
  rung is read in a pilot on worlds held out of M3's confirmatory run, under a rule written before
  the pilot, and the S-shaped families are reported beside the claim's.
- A kernel of 6 weeks fits in less than half the CPU time and is not a candidate: at the length
  the worlds are drawn with, two concave worlds' plans lost what the panel's kernel kept.

## Alternatives considered

- **The `tanh` always.** Planning from the `tanh` removed most of the gap where the Hill had taken
  the weight, and added 0.139 a euro where the truth is S-shaped: it is the confirmatory families'
  own generator, not a rule.
- **A mixture of the two families' plans**, by half or by the stacking weight: it recovered none of
  the change in M2's refits.
- **Lift tests at a second spend level**, which read the margin and not only the chord: a change to
  the bench's design, not the engine's.
- **A second stage with a job of its own** (the DLM read without the tests, as a check of drift):
  it costs a DLM fit on every world for a reading the claim does not make.
