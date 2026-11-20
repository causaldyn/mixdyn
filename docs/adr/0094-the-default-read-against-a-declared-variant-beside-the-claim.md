# 0094. The default read against a declared variant beside the claim

Status: proposed, 2026-11-17; adds `Declaration.default_less` and `scorecard.claim.against_default`
with `stratified`; changes no comparison, verdict, Holm's list or headline, and a declaration
that names no variant is written as before.

## Context

A run's claim compares its default with every rival. The engine's other variants are not rivals:
`recorded` reads them against the rivals, never against the default. A run may still need one
such reading. A claim's procedure chosen after a pilot replaces the pilot's own winner, and the
fair check of that choice is the new procedure against the old one on worlds no one has read.
That reading is neither a rival's comparison, which would enter the intersection-union test,
the fixed sequence and Holm's list, nor a reading after the run, which would not be declared
before it.

## Decision

- **Declared with the run.** `Declaration.default_less` names variants of the run's own arms. It
  is refused on a run the claim does not read, on one without rung 0, twice, or where it names the
  default or a tool's arm. It is one of the fields a pool's runs declare alike (`AGREED`).
- **Read beside the claim.** `against_default`, printed after the secondary readings, reads the
  default at rung 0 less each named variant, run by run, as `recorded` reads an arm: on the
  complete cases, with the paired 95 % Student-t interval and each arm's failures.
- **Pooled over the confirmatory runs.** For each default and variant, every complete case of the
  runs on confirmatory worlds counts as one world. The mean's variance is stratified by run,
  `sum_f n_f s_f^2 / N^2` (`stratified`), and the 95 % interval is on the normal quantile.

## Consequences

- The reading enters no verdict, no Holm's list and no headline.
- A variant's arm runs at every rung of its root, since a root's arms run at all its rungs and a
  root beside a run must hold every rung of the run (`read_beside`). Records above rung 0 are
  reported by `recorded` and are not read against the default.
- The stratified variance leaves out the spread between runs' means, which the design fixes by
  declaring each run's worlds. At a thousand complete cases or more the normal quantile is within
  0.2 % of Student's.

## Alternatives considered

- **The variant as a rival.** It would enter the intersection-union test and Holm's list. Both
  are sized and ordered on the rivals' predictives, and the variant has none.
- **One Student-t interval over every pooled world.** It is valid but wider. It counts the
  spread between the runs' means as noise, though no run's share of worlds is random.
- **Each run weighted equally.** It answers another question, the mean over families. The
  per-run lines already give each family's reading.
