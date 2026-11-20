# 0084. Tuned rival arms, and the rivals in the refresh cell

Status: proposed, 2026-11-16; adds `scorecard.tuning`, three tuned tools' arms
(`scorecard.tools.TOOLS`: `meridian tuned`, `pymc-marketing tuned`, `robyn tuned`) and the
bench's options their scripts take; the refresh cell fits the tools' arms and runs from a plan
(ADR 0079, amended 2026-11-16); amends ADR 0039; no default arm, no default command line and no
existing record changes

## Context

M2 ran each rival at its recommended settings. A reader can fairly object that a practitioner
tunes a tool before trusting it, and that a headline against untuned rivals may move once they are
tuned. A tuning chosen after the scored worlds are read is a forking path, so a tuned arm has to
be fixed by a written rule, on worlds the scored run does not read, before any scored world.

Each rival states its own criterion for a configuration. Meridian's model reviewer gives a health
card (an overall status, a health score, and the goodness-of-fit check on held-out weeks where the
model holds some out). PyMC-Marketing ships a time-slice cross-validator. Robyn validates on a
time-series split of its own (`ts_validation`). The tuning reads each rival by its own criterion,
never by the bench's score or a world's truth.

A refit's stability is also what vendors sell. The refresh cell (ADR 0079) read this engine's arms
alone.

## Decision

**Three tuned arms, each recorded apart** from its rival's recommended arm, so that the
recommended arm's records stay comparable with M2's.

| arm | candidates | chosen by |
|---|---|---|
| `meridian tuned` | the media prior type (`roi`, `mroi`, `contribution`) by the knots (automatic selection; knots at most 8 or 13 weeks apart): 9 | Meridian's health card, then its fit on held-out weeks |
| `pymc-marketing tuned` | the saturation and the intercept: logistic static, logistic varying, Hill static: 3 | the library's time-slice cross-validation |
| `robyn tuned` | none: the genre ranges with `ts_validation` on and the tests as `calibration_input` | the rule itself; families 9, 12 and 13 only |

The first candidate of each grid is the rival's recommended settings (`scorecard.tools.GRIDS`,
`RECOMMENDED`).

**The worlds.** The development set is M2's scored worlds. Within each family and environment, M2's
seeds sorted ascending, a world whose rank mod 3 is 2 is a check world, and so is every world of a
root that holds Robyn; every other world is a fit world. The tuning reads the first five fit
worlds of each of M2's 28 national environments at rung 0: 140 worlds. A configuration is a model
choice made before any test is read; the tuned arm reads each rung's tests as its rival's arm does.

**Meridian's rule.** Every candidate is fitted at the arm's sampler with the last tenth of weeks
held out of the likelihood. Admissible: a share of fits whose overall status is not `FAIL` within
0.05 of the best candidate's, and a mean health score within 1.0 of the best's (a fit that raised
counts as `FAIL`, score 0). Among them, the lowest mean `wmape_test` on the held weeks wins, over
the worlds every admissible candidate shares.

**PyMC-Marketing's rule.** Every candidate's cross-validation: quarterly folds after a history
that grows by a quarter, the last fold ending on the last week (four folds at 156 weeks), at half
the arm's draws; the score is the folds' mean test CRPS in units of the history's mean weekly
sales. A candidate whose cross-validation raised on more than 5 % of the worlds is out. Then the
mixing gate, as Meridian's card gates Meridian: a world mixed where every fold's largest R-hat is
at most 1.1 and its divergent draws at most a tenth of its kept draws (the arm's own flags); a
candidate whose share of mixed worlds lies more than 0.05 below the best candidate's is out. Among
the rest, the lowest mean CRPS wins.

**Ties, for both.** Every candidate whose paired mean difference from the winner is within one
paired standard error ties with it. A tie goes to the recommended settings, then to the candidate
that moves fewer settings, then to the lower mean.

**The choice is committed before the freeze.** `scorecard.tuning select --write` writes it to
`scorecard/tuned.json`, with each candidate's summary, why each loser lost, and a SHA-256 of every
record read. Until that file names a rival's choice, its tuned arm is refused where a run declares
it.

**A cost cap, alike for every rival.** A candidate fits within 2 times its rival's recommended
settings' CPU a fit (`scorecard.tools.COST_CAP`), so the tuning searches only what a scored run
can afford. PyMC-Marketing's Hill saturation with a varying intercept fitted at 7.7 times on one
world and left the grid; every other candidate measured or bounded lies below 2.

**The rivals in the refresh cell.** Each cut of a world is exported as the bench archives a world,
and the tool's script runs on it as the runner runs a tool's world (ADR 0079's amendment of
2026-11-16). A whole cell runs from one plan, each cut a process of its own. For M3 the cell holds
family 0's two environments and family 1's walk and static worlds as swings, and family 7's
104-week window as a rolling origin; 10 pilot worlds an environment, at rung 0; every arm, the
tuned rivals among them, but Robyn.

## Consequences

- **Measured at full samplers** on one fit world of 156 weeks: Meridian's candidates 220 to 340
  CPU seconds a fit (M2's recommended fit of the same world, 293); PyMC-Marketing's 169 (its
  recommended settings), 270 and 283; a cross-validation 352 CPU seconds at the recommended
  settings; Robyn's tuned arm 12 299 CPU seconds on a family 13 world at rung 4, where M2's
  recommended fits at rungs above 0 took 10 026 to 11 368.
- **The tuning costs** about 99 CPU-hours for Meridian (1260 fits) and 72 for PyMC-Marketing (420
  cross-validations), at M2's per-family costs.
- **Where the recommended settings win,** the recommended column is the tuned one: a run that
  declares the tuned arm is refused (`scorecard.tools.recommended`, amended 2026-11-17), since its
  column would fit the recommended column again.
- **Limitations, stated with the claim.** The tuning reads national worlds at rung 0, and its
  choice holds at every rung and on the geo families. It searches a small grid within the cost
  cap. Meridian fits a tested channel at the `roi` prior whatever type was chosen, since it takes
  an experiment through that prior alone.

## Alternatives considered

- **Meridian's random holdout.** Weeks drawn at random sit between weeks the fit reads, and an
  autocorrelated series leaks through them to the time effects (ADR 0079 measured the leak), so it
  favours the most knots. The last weeks are a forecast, as the cross-validation reads one.
- **Meridian's automatic knot selection over every week.** The selection reads the outcome of every
  week it is handed, held weeks too: on one world 5 of its 40 knots fell inside the 16 held weeks.
  The tuning's fit selects knots on the fitted weeks alone; a fit without a holdout is unchanged.
- **A mixing gate on a fixed share.** 5.6 % of M2's fits of PyMC-Marketing's recommended settings
  on its national worlds at rung 0 are past R-hat 1.1, and every one reports divergent draws, so a
  gate of 5 % of worlds would hold out the recommended settings themselves.
- **No cost cap.** A Hill saturation with a varying intercept could be chosen and multiply the
  scored run's PyMC-Marketing compute by about 7.7.
- **Tuning each family apart.** It costs a grid a family and reads each family's own worlds twice;
  one configuration a rival is what a practitioner carries to a new brand.
- **Tuning Robyn's ranges.** The genre and union ranges were piloted before M2, and the genre ranges
  won; the tuned arm moves the one setting left, its time-series validation.
- **Robyn in the refresh cell.** 7.8 to 8.5 CPU-hours a world for its seven fits, at M2's costs a
  fit on families 0, 1 and 7; it runs anew only on families 9, 12 and 13.
