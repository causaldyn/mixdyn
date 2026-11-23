# 0100. A rung past the cap is exploratory, not predicted to fail

Status: accepted, 2026-11-23; adds `Declaration.exploratory_rungs`; a declaration that does not set
it is written and read as before.

## Context

A family and rung whose worlds needed pass the sizing cap is run, read and printed, and left out of
the claim: it is exploratory. A declaration could say so only of a whole run (`exploratory`). M4's
k = 0 rungs of families 1, 4, 6 and 14 were past the cap while their tested rungs were claimed, so
M4's roots kept those rungs out of the fixed sequence by naming every rival there in `predicted`.
The reading then printed "predicted to fail" for families 1, 6 and 14, where the pre-registration
predicts no failure, and Holm's step-down counted their 15 comparisons, with family 4's five, among
the claim's. A predicted failure is a forecast about a comparison; a rung past the cap is a fact of
the sizing. They need different words and different places in the reading.

## Decision

- **A run declares its rungs past the cap** in `exploratory_rungs`, a rung above 0 named as a
  pool's first run names it, as `predicted` and `order` name it.
- **Such a rung reads as an exploratory run does.** Each of its comparisons is exploratory: printed
  with its interval and word, left out of its verdict, Holm's list and the order of verdicts, which
  is refused where it names the rung.
- **A family and rung read only as exploratory stops the headline.** The headline needs every
  family and rung claimed whole (the pre-registrations of M3 and M4 say so), and the reader now
  prints each such family and rung beside the verdicts. A tool read on its cap and exploratory is
  read beside a verdict that holds and does not stop it, as before.
- **`predicted` keeps its meaning**: comparisons the pre-registration forecasts to fail, read and
  printed, left out of their verdict and kept in Holm's list.

## Consequences

- M4's roots declare `predicted`, so their reading is unchanged. A later run declares its rungs past
  the cap here and keeps `predicted` for forecasts.
- An exploratory run's family no longer leaves the headline standing.

## Alternatives considered

- **Reading every rung whose rivals are all predicted as exploratory.** Rejected: a rung can be
  past the cap and also hold a forecast failure, as family 4 at k = 0 in M4 does, and the reader
  would have to guess which was meant.
- **Overriding the declaration from the command line when reading.** Rejected: it would let a
  reader move comparisons out of the claim after the data are seen.
