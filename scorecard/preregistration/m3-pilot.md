# The M3 pilot: pre-registration

**Committed before the pilot**, every slot filled. The pilot's roots declare this file as the
pre-registration they run under (`Declaration.preregistration`, its SHA-256 stamped), and it does
not move while they run: an arm of a root whose file moved does not start. The selection rule below was
declared on 2026-11-16, before any M3 pilot ran. M3's own pre-registration (`m3.md`) cites this
file and is frozen after the pilot.

## What the pilot runs

- **The development roots**: the candidate arms on M2's scored worlds, M3's development set, one
  root a main M2 root and rung kind: the k = 0 candidates at rung 0, the tested-rung candidates at
  the tested rung, the geo candidates on families 8 and 10 at 3 and 8 geos. They hold new arms
  only. Each is a scored-kind declaration at M2's seeds (`pilot` false), at the bench's commit
  `67f92b7478b0a1423538d77e62d8198923c52a85` and mixdyn's commit that holds this file with every
  slot filled, which each root's `run.json` names.
- **The pilot roots of families 9, 12 and 13**, which M2 never ran, on their pilot seeds:
  30 worlds an environment at rungs 0 and 4, one root a family, each declared with the margin
  0.05 and `pilot` true, so that `claim read` accepts it and `select --fresh` reads it for the
  winners' predictive: families 12 and 13 (`priors`) the arms `dlm`, `bayes`, `bayes-concave`,
  `ridge` and `chc`, family 9 (`spillover-0.2`, `spillover-0.5`) the arms `bayes-geo-plug-in`,
  `geo-dlm`, `ridge` and `chc`, each beside the tools Meridian, PyMC-Marketing and Orbit at their
  recommended settings (no tuned tool: the pilot's tuning writes `scorecard/tuned.json`); and Robyn
  at its recommended setting and tuned on the first 3 worlds an environment of each of the three
  families at rungs 0 and 4, beside `chc` (a smoke, declared without a margin, not read by
  `select`).
- **The rivals on M2's worlds are not fitted again.** Their stored records are read beside the
  development roots (`scorecard.development.joined`), or taken into a root
  (`python -m scorecard.run reuse`), only where the world's export is the observation the record
  names by digest and the tool's pinned files are unchanged, or changed only outside its default
  setting and accepted by name: the bench's `scripts/meridian_arm.py`,
  `scripts/pymc_marketing_arm.py` and `scripts/robyn_arm.R`. Each changed between M2's bench
  commit `316fada` and the pilot's only by options a tuned arm passes (the media prior type, the
  knots and a holdout; the saturation and the intercept; `--ts-validation`), and with no option
  each builds the model it built before (the diff). The pilot checks it before any development
  root is read: the first two fit-split worlds of family 0 `drawn` (seeds 10021 and 10022) at
  k = 0 refitted by Meridian and by PyMC-Marketing, and the world of seed 10000 of M2's `f0-robyn`
  root by Robyn, at their recommended settings with the pilot's scripts, each weekly plan printed
  against the stored one. A tool whose replayed plan
  moves any channel's spend by more than 10⁻⁶ of the budget is not reused into any M3 root; its
  stored records are still read for the selection, as the plans that tool made at M2's commit.
- **The candidates** (each a recorded variant the runner produces, at most four a rung):
  k = 0 `dlm cvar 0.1` and `dlm cvar 0.1 vs equal`, both M2's recorded variants of the `dlm`
  arm, read from its records; tested rung `stacked`, M2's recorded variant of the `bayes` arm, and
  `stacked, concave hill` and `stacked, concave hill, lift-row pick`, the two variants of the
  `bayes-concave` arm from one fit; geo `bayes geo plug-in`, M2's recorded variant of the
  `bayes-geo` arm, and `geo dlm`, the `geo-dlm` arm; tuned rivals Meridian's nine configurations
  (the media prior type `roi`, `mroi` or `contribution`, by the knots AKS places or knots at most 8
  or 13 weeks apart; `roi` with AKS is the recommended one) and PyMC-Marketing's three specs
  (`logistic static`, the recommended one, `hill static`, `logistic varying`; `hill varying` fits at
  7.7 times the recommended fit's CPU, past `scorecard.tools.COST_CAP`, 2), each rival's chosen by
  its own criterion, never the truth (`scorecard.tuning`), on the first 5 fit-split worlds of each
  national environment at k = 0. The candidates file names each
  candidate's rung, its settings changed from its arm's defaults, and its CPU-seconds a fit where
  the records do not state them.

## The development set and its split

- M2's scored worlds, every root of M2's run, with every rival's stored records.
- Within each family and environment, M2's scored worlds are ranked by their seeds, ascending. A
  world whose rank leaves 2 when divided by 3 is in the `check` split, and so is every world of a
  `<root>-robyn` root. Every other world is in the `fit` split (`scorecard.development.split`).
- Candidates are tuned, selected and compared on the fit split alone. The check split is read once
  a candidate rule, for the predictive.

## The selection rule

1. Data: the fit split, M2's stored rival records, each pilot world paired with M2's world of the
   same family, environment, seed and rung only where both exports hold the same observation by
   digest and the built-in arms score alike.
2. For each candidate and each confirmatory family f at rung r: `d(f, r, rival)`, the mean paired
   difference of regret per euro, the candidate less the rival, over the fit split's worlds both
   were read on. A fit that failed plays the status quo, as the bench scores it, and is counted;
   no world is dropped for a failure, and each candidate's failures on each family are printed. A
   candidate's robustness is part of its quality; the predictive reads complete cases, as the
   claim does (below).
3. A candidate's score at rung r is the largest `d(f, r, rival)` over the confirmatory families
   and rivals. A rung is `0`, or `tested`, every rung above 0 a family's worlds are read at. The
   confirmatory families: 0, 2, 3, 5 and 7. Families 9, 12 and 13 hold no development world and
   are read for the predictive alone (below).
4. The candidate with the smallest score wins rung r. Candidates within 0.005 of the smallest tie;
   a tie goes to the fewest CPU-seconds a fit, then to the fewest settings, then to the name.
   Before that, at the tested rung, the S-curve guard: families 4 and 14, whose truth is a Hill
   curve, are not in the score, but a candidate whose mean regret per euro on the fit split of
   either, at the tested rung, exceeds `stacked`'s by more than 0.02 is out
   (`scorecard.development.guard`); the pilot plans every tested-rung candidate on those two
   families' worlds for it.
5. The geo families, 8 and 10 at 3 and 8 geos, are scored apart: the rule runs over those two
   families alone, with the geo candidates.
6. The winner's predictive is computed on the check split, never the fit.

`python -m scorecard.development select --stored <M2's roots> --pilot <the development roots>
--fresh <the pilot roots of families 9, 12 and 13> --candidates <file> --worlds <family>=<N> ...
--resamples 4000 --seed 20261116` applies the rule and prints the S-curve guard, the winner of each
rung, its ties, its table of `d`, what common random numbers take off each of its comparisons, and
its predictive on each family, families 9, 12 and 13 read on their pilot worlds, every one in the
check split.

## The predictive

For each comparison the winner would make on confirmatory worlds
(`scorecard.development.predictive`): in each environment, the check split's worlds are drawn with
replacement, as many as it holds, which carries the pilot mean's own error; from those, as many as
the confirmatory run will read there (the sizing's N), drawn with replacement; each draw is read
as M3's claim reads it, each comparison on its complete cases, a world where either arm failed
drawn and left out of it, by Hall's reader against δ = 0.05 at one-sided 0.025. A
comparison's predictive is the share of draws that read it within; a family's at a rung is the
share that read every claimed comparison within together, and it sets the family and rung's place
in the order M3's verdicts are tested in, highest first, ties by family, then rung. A comparison
whose predictive is below 0.5 is a predicted failure: run, read, printed and left out of its
verdict. 4 000 resamples, seed 20261116.

## What else the pilot measures before M3's pre-registration is frozen

- **The switchback reader's criterion.** If M3's pre-registration states one, it is the coverage
  of the reader's nominal 90 % interval pooled over worlds, with its binomial interval, beside the
  pooled error in standard errors; never one world's relative error. Its threshold is fixed from
  a coverage pilot: the `switchback` arm on the first 30 of M2's scored worlds of family 8
  `policy-0`, at the tested rung (the arm reads the same switchback at every rung). The coverage
  c is the share of the 30 whose interval, a week on, holds the steady truth (the score's `held`),
  with its Clopper–Pearson 95 % interval [l, u]. If u is below 0.80 the reader under-covers in
  plain sight: M3 states no switchback criterion and prints its coverage as exploratory, said to
  under-cover. Otherwise the threshold is the smaller of 0.85 and l, rounded down to a multiple of
  0.05, and M3's criterion holds if the upper end of the Clopper–Pearson 95 % interval of its own
  switchback worlds' coverage reaches it.
- **Common random numbers and control variates.** Their variance reduction is measured on the
  pilot before N is fixed. Every arm reads the same worlds, the same noise and the same scoring
  draws: the bench's paired design is M3's common random numbers. `select` prints, for each of the
  winner's comparisons on each family's fit split, the variance of the paired difference over the
  two arms' variances summed on the same worlds, the variance independent worlds would give. M3
  uses no control variate. N is set from the variance the pilot measures with the reduction M3
  uses: the paired one.
- **Cost.** Each process's CPU-seconds, wall and peak memory (`logs/costs.jsonl`) and each attempt
  (`logs/attempts.jsonl`), which M3's compute and limits are set from.

## Stop rules

- No stop on an outcome: every development world of a candidate is read before the rule is
  applied.
- A defect found in a candidate, a tool's configuration or the harness: the candidate stops, the
  fix is committed, and its worlds are fitted again; the pilot's report names it.
- The pilot's budget: about 990 CPU-hours (988 by the stage's costing, the replay check's 1.5
  among them), on one c6a.48xlarge spot instance for about 8.5 hours (makespan median 7.0 h, the
  walls' 90th percentile 8.7 h): $10.6 at the median and $14.2 at the 90th percentile of the
  costing. Past $20 or 1 400 CPU-hours the pilot stops and the author decides; the stop reads no
  outcome.
