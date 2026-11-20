# 0073. Twin worlds a history cannot tell apart, and a seesaw test that can

Status: accepted, 2026-11-06; adds `mixdyn.experiments.seesaw` and the twin worlds of
`tests/twin_worlds.py`; changes no default and no existing output

## Context

Dew, Padilla and Shchetkina (arXiv 2408.07678) find that a response that saturates and an effect
that drifts over time explain the same history: in most of their simulated settings and in all six
of their real datasets, a held-out fit does not tell the two apart, and the budgets the two imply
differ. Where a channel's spend rose over the history, a curve that rises with the spend and a
line whose slope rises with the weeks put the same sales in each week. Their remedy is a test that
moves the spend up and down within the same weeks, a seesaw.

The engine fits both readings. The Bayesian model's curve families (ADR 0011) read a fixed curve,
and the DLM (ADR 0020) a coefficient that drifts. The bench scores each on worlds built for one of
them, the drift family and the S-curve family apart, so no world asks an engine to choose between
them. The candidate tests (ADR 0030) multiply a channel's spend by one factor over a stretch of
weeks, and the switchback (ADR 0056) turns a national channel on and off.

## Decision

- **The twins** (`tests/twin_worlds.py`): one history, two truths. Television's spend rises from 30
  to 100 a week over 78 weeks and holds there for 26 more, each week's spend its level times a
  log-normal of log-scale 0.05; search's is drawn about 50 a week. In `"curve"`, television's
  response is fixed on a Hill, `350 x^3 / (x^3 + 60^3)`. In `"line"`, it is `drift(t) x`, the slope
  the curve's average return at the week's level: 1.30 at the first week, 3.09 at most, and 2.88
  from week 77. Search's response, `252 tanh(x / 50)`, the base and the noise, 40 a week, are the
  same draw in both. Neither world carries spend over. A plan spends 150 a week for 13 weeks,
  television within `[50, 140]` and search the rest. The line's plan reads the slope where the
  history left it.
- **`seesaw(panel, geo, channels, noise, *, when, high=1.5, low=0.5, block=2, blocks=8,
  cooldown=8, shares=...)`** in `mixdyn.experiments`. Each channel's planned spend is multiplied
  by `high` and `low` in turn, `block` periods each, over an even number of blocks from a high
  one, then a cooldown at the plan, in a group of each share. It returns `Candidate`s laid out as
  `designs` lays out a test (ADR 0030), so `predicted_lift`, `value_of_tests` and `choose_tests`
  read them as any other test.
- **Why this form.** The twins agree at the plan's spend, where the line's slope is the curve's
  average return, and at none, where both return nothing. A test that compares the plan with none
  reads only what they share. What differs is the response's slope on either side of the plan:
  under the line, a unit above the plan returns what a unit below it does, and under the curve, a
  unit above it returns less. A seesaw reads both sides, at the plan's own spend in sum:
  - under any line its high and low blocks' gaps are equal and opposite, so its summed lift is 0
    whatever the slope, and under a curve that bends at the plan it is not;
  - read period by period (`LiftSeries`, ADR 0046), each high block reads the chord above the plan
    and each low block the chord below it;
  - alternating blocks keep both levels within weeks a slow drift moves alike. One stretch high
    and then one low would read the drift between them as curvature;
  - `high` and `low` symmetric about 1 spend what the plan spends; 1.5 and 0.5 read the curve near
    the plan, where the decision is.
- **The bench gains the family later.** The worlds here are the engine's tests. The bench is
  another repository, and this change edits none of it.

## Consequences

- The twins' best plans put television at 110.81 a week (line) and 94.69 (curve). The line's plan
  costs the curve's world 0.095 a euro of the budget, and the curve's plan costs the line's
  0.061.
- A ridge fitted to the first 78 weeks, television on a Hill of slope 3 and search on one of
  slope 1, scores the 26 held-out weeks alike in both twins: the paired difference of the weekly
  log scores holds 0 in its 95 % interval on all 20 seeds from 20261106 to 20261125.
- **Every engine plans the twins alike** on their whole history, seed 20261106: television a week
  in each twin's plan, and that plan's regret a euro in its own twin.

  | engine | curve's twin | line's twin | regret, curve | regret, line |
  | --- | --- | --- | --- | --- |
  | ridge | 98.68 | 98.69 | 0.006 | 0.035 |
  | Bayesian, Hill | 100.34 | 102.54 | 0.012 | 0.017 |
  | Bayesian, tanh | 98.06 | 98.58 | 0.004 | 0.036 |
  | the two stacked | 98.06 | 98.58 | 0.004 | 0.036 |
  | DLM, its coefficient drifting on the spend | 120.33 | 122.51 | 0.239 | 0.035 |
  | the plan in place | 100 | 100 | 0.010 | 0.028 |

  The models of a fixed curve plan near the plan in place in both twins and pay up to 0.036 in
  the line's; the DLM, a line, plans past both twins' best and pays 0.24 in the curve's. Within
  each engine the two plans differ by at most 2.2 a week, while the twins' best plans lie 16.1
  apart: the history does not tell an engine which twin it is in.
- **The seesaw separates them.** In a group of 0.3 of the market, after the history, with the
  plan at television 100 and search 50, and an 8-week cooldown, the likelihood ratio between the
  twins on the readout's weekly gaps picks the twin that drew them with probability `Phi(d / 2)`,
  `d` the distance between their expected gaps in the gaps' noise:
  - the default seesaw, 2-week blocks at 1.5 and 0.5 times the plan over 16 weeks: 0.954;
  - the same over 32 weeks: 0.991;
  - 2-week blocks at twice the plan and none over 16 weeks: 0.9999;
  - a go-dark of 16 weeks: 0.5, no better than a coin, as its expected gaps are the same in both.
  A heavy-up to twice the plan for 16 weeks separates them further, `d` = 10.76, and spends 480
  more in the group, where the seesaw spends nothing more.
- **The tests** (`tests/test_twin_worlds.py`, `tests/test_experiments.py`):
  - the regret gap above 0.05 a euro each way, and each twin's best plan of no regret in its own
    world;
  - the ridge's held-out tie on seed 20261106;
  - each engine's television plan within 5 a week alike in both twins;
  - the go-dark's gaps alike in both twins to 1e-12, and their distance below 1e-9;
  - a seesaw's summed lift 0 under the line, below 0 under the curve, its spend the plan's; the
    rates above 0.99 and 0.95; and the default's rate within 4 Monte Carlo standard errors of the
    share picked rightly over 4000 readings drawn in each twin;
  - the heavy-up's distance at least the seesaw's and its 480 more;
  - `predicted_lift` on the curve's twin as one reading equal to the curve's summed gaps;
  - the seesaw's layout, its labels, readout and standard errors, and the inputs it refuses.

## Alternatives considered

- **A go-dark, `designs`' multiplier 0.** It compares the plan's spend with none, the two points
  where the twins agree by construction. It reads the average return they share and separates
  them not at all (a test, exactly).
- **The switchback (ADR 0056).** It turns a national channel on and off, a go-dark repeated, and
  reads the same average return.
- **A heavy-up, `designs`' multiplier 2.** It separates the twins further, since it reads the
  curve's flattening in every week where the seesaw at twice the plan reads it in half of them.
  It adds the spend it doubles, 480 in the group over 16 weeks here. The seesaw spends what the
  plan spends. Both stay candidates: `choose_tests` prices each.
- **Twins drawn with noise of their own.** Each twin's own noise would make the held-out tie a
  property of the draw. Sharing it makes the gap between the twins' scores the gap between their
  truths alone.
- **The seesaw read by its summed lift alone**, as `value_of_tests` reads a candidate. Under any
  line it reads 0 and under a curve its bend: a test of the line that needs no slope, but it reads
  the curve's second derivative, not its slope on either side. Read period by period, the blocks
  read the slope.
- **Moving the family into the bench now.** The bench is another repository, and this change
  edits none of it.
