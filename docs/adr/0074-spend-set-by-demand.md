# 0074. Spend set by demand: a demand proxy's one role, and the endogenous-spend flag

Status: proposed, 2026-11-06; adds `mixdyn.models.demand`, `mixdyn.data.feedback` and the issue
code `ENDOGENOUS_SPEND`; no default and no existing output changes

## Context

A media-mix model reads a channel's return as identified when nothing it leaves out moves both the
channel's spend and the KPI. Two common ways a business spends break that.

- **Paid search buys against demand.** Its ad shows where someone searches, so its spend rises
  with the queries, and the queries move the KPI on their own. Chen et al. (2018, "Bias Correction
  For Paid Search In Media Mix Modeling", Google) found naive search returns 3 to 15 times what
  geo experiments read, and a category query-volume control removed part of it. Meridian's
  documentation reads Google query volume as a confounder for paid search and a mediator for the
  other media: television raises the searches for the brand. One model that holds the queries
  fixed blocks television's path through them; one that leaves them out credits search with the
  demand. It must pick one role.
- **The budget reads the sales.** Heusch (2026, "A Synthetic Benchmark Dataset with Endogenous
  Marketing Spend for Validating Marketing Mix Models", arXiv:2608.21130) builds a retailer whose
  quarterly budget follows the last two quarters' sales before media, whose paid shopping is bid
  up after a week above its ten-week average, and whose baseline carries a sentiment nobody
  observes. On it an observational model with realistic controls read paid shopping's return at
  10.66 against a true 4.20 (his companion paper, arXiv:2608.21128). The bench implements the
  world clean-room as Track M v2 (`causaldyn_bench.endogenous_mmm`).

PyMC-Marketing's open proposal of a budget model (issue 3055, 2026-09-25) writes the spend as a
predictable part and a surprise, `spend = m + v`, and adds `gamma v` to the KPI's mean: a `gamma`
clearly off 0 flags demand-chasing spend. mixdyn had no demand control and no such flag.

## Decision

- **`DemandProxy(column, role, searched)`** (`mixdyn.models.demand`) names a panel's control that
  measures demand, the channels whose spend follows it, and the role this model gives it, with no
  default. `controls()` is what a spec reads: in the `"confounder"` role the column as a control
  held at or above 0, demand raising the KPI; in the `"mediator"` role nothing. `reading(channel)`
  states what a channel's return then is: `"adjusted"` for a searched channel and `"direct"` for
  any other in the confounder's role, `"confounded"` and `"total"` in the mediator's. The proxy is
  a helper over the specs' controls, so `RidgeSpec`, `BayesSpec` and the save format are
  unchanged.
- **Both readings at once are the nested model's** (ADR 0014): stage one reads the proxy on the
  other media, its prior on each return stated in the proxy's units, so their path through it is
  counted, and stage two reads the KPI with the proxy a control, the confounder's model. One stage cannot hold both roles; two can, under the nested
  model's own assumption that nothing unobserved moves the proxy and the KPI together beyond
  stage two's calendar and controls.
- **`spend_feedback(panel, geo, level=0.95, harmonics=3)`** (`mixdyn.data.feedback`) reads each
  channel's spend as the business's budget model and its surprise, `spend_t = m_t + v_t`, with

      m_t = a + phi spend_{t-1} + gamma kpi_{t-1} + b t / n + season_t,

  the season `harmonics` harmonics of a year at the panel's cadence, over the periods observed with
  the one before. `gamma` is the spend per unit of last period's KPI, of either sign. Its standard
  error is Newey and West's, Bartlett weights `1 - j / M` to `M - 1 = floor(4 (n / 100)^(2/9))`
  lags counted on the calendar, so a gap holds no pair, each residual scaled by one over one less
  its leverage, as HC3 scales it. Its interval's half-width is that error times Kiefer and
  Vogelsang's (2005) fixed-b critical value for the Bartlett kernel at `b = M / n`, their cubic in
  `b`, tabled at the levels 0.8, 0.9, 0.95 and 0.98; a simulation of the statistic they
  approximate, at `b` from 0.02 to 0.2, matches each cubic within 2 %. Where the interval excludes
  0, `SpendFeedback.issue()` is an `ENDOGENOUS_SPEND` warning naming the geo and the channel.
  - *The last KPI*, because that is Heusch's construction: his budget and his bidding rule read
    past sales before media, which the KPI carries. Where the unobserved demand persists, the past
    KPI carries the present demand into the spend.
  - *The spend's own last period*, because the media's carryover ties the last KPI to past spend:
    a flighted or slowly planned spend that reads nothing of the KPI would otherwise read as one
    that follows it.
  - *The trend and the season*, because a plan that follows the calendar, as his seasonal tilt
    does, is set by something both series share and the model holds.
  - *Level 0.95*, a test's usual size, so that an exogenous spend is flagged in about one seed in
    twenty; the engine's returns keep their 0.9.
  - *The leverage and the fixed-b value*, because without them the interval was too narrow: over
    1000 seeds of Track M v2's exogenous worlds Student's interval on the unscaled error, `n / (n -
    k)` its only correction, flagged 7.3 % to 8.0 % of a channel's spends at 0.95. A promotion's
    week puts the last KPI far from the rest, and an error read without its leverage reads it too
    small. With both, 5.3 % to 5.7 %.
- **Not the proposal's `gamma`.** It sits on the surprise `v` in the KPI's equation, beside the
  channel's own term, which is a function of `m + v`. It is told apart from the channel's effect
  only through what moves `m` and not the KPI, a variable a panel rarely holds, and it reads what
  the surprise holds of the KPI's unobserved part, which depends on what `m` reads. Where `m` reads
  the past sales, as Heusch's plan does, the chasing is in `m`, the surprise is the plan's own
  noise, and `gamma` reads 0 however hard the spend chases. The flag reads the plan's dependence
  on the KPI itself, and needs no such variable.
- **Opt-in.** `read_panel` does not run it: a regression on every panel read would change
  `MediaPanel.issues` for every caller. The code joins version 1 of the issues' document
  (`mixdyn.api.ISSUES`), as `CADENCE` and `FEW_OBSERVATIONS` did, since no engine that read
  version 1 has been released.

## What each reading means, and what it does not

- **The flag is not a correction.** It says the spend reads the outcome, which is what lets a
  persistent unobserved demand bias a return. It says nothing of how far a return is biased;
  `mixdyn.inference.sensitivity` (ADR 0061) bounds that.
- **A spend that reads the demand of the same period**, as search's auction volume does, shows in
  the flag only as far as the demand persists. A demand proxy in the model is its remedy, and the
  proxy holds the confounding only as far as it measures the demand.

## Consequences

- *Tests*: `tests/test_feedback.py`, `tests/test_demand.py`, and a fixture of Track M v2's worlds,
  `tests/fixtures/track_m2_worlds.json` (five seeds each of his reference instance and of the
  generator with its four mechanisms off, the observed panel and each channel's true return, the
  bench's version and call in its provenance), so that the tests reading it run in CI without the
  bench. A test checks the fixture is the bench's where the bench imports.
- *The search-demand world* (`tests/search_demand_world.py`) is built in mixdyn's tests: an organic
  demand with a hidden AR(1), television that raises it, queries that measure it within 2 %, and
  search spend that follows it, in the Bayesian model's own form.
- *The study* (`scripts/study_endogenous_spend.py`, its runs in its docstring) measured:

  - **The flag**, at its defaults, 1000 seeds of each world; each cell the share of seeds whose
    interval excludes 0, its Monte Carlo standard error at most 0.016:

    | world | pla | meta | tv | search | any channel |
    |---|---|---|---|---|---|
    | his reference instance | 0.648 | 1.000 | 0.001 | | 1.000 |
    | the four mechanisms off | 0.057 | 0.056 | 0.053 | | 0.159 |
    | budget feedback alone | 0.047 | 0.040 | 0.043 | | 0.124 |
    | anticipatory spend alone | 1.000 | 1.000 | 0.052 | | 1.000 |
    | television's bursts alone | 0.055 | 0.054 | 0.000 | | 0.107 |
    | the bidding rule alone | 1.000 | 0.054 | 0.052 | | 1.000 |
    | the search-demand world | | | 0.055 | 0.858 | 0.866 |

    At his reference instance the flag fires at every seed. It reads meta's anticipatory spend at
    every seed, its `gamma` below 0: spend raised ahead of a promotion and held back in it reads a
    strong week with a fall. It reads paid shopping at 648 seeds of 1000, its bidding rule pushing
    `gamma` up and its anticipation pulling it down; the bidding rule alone it reads at every seed.
    It does not read the quarterly budget's feedback alone: a multiplier held over 13 weeks from
    two quarters' sales is not in one week's KPI beside the spend's own last week.
    So the criterion, `gamma`'s interval excluding 0 at the reference parameters, is met for the
    world at every seed and for paid shopping at 65 % of them; covering 0 on exogenous worlds in
    90 % of seeds is met channel by channel, 94.3 % to 94.7 %. Read over a world's three channels
    together it is not: some channel's interval excludes 0 at 15.9 % of the exogenous seeds, as
    three tests at 0.95 do. At `level=0.98`, read from the same 1000 seeds' `gamma` and error,
    some channel's interval excludes 0 at 6.5 % of the exogenous seeds, each channel's at 2.1 %
    to 2.6 %, and paid shopping's at 37.7 % of the reference seeds, meta's still at all.
  - **The demand proxy**, on the search-demand world at 10 seeds, the one-geo Bayesian model with
    the queries in each role. Each return's posterior median over its truth, the median and the
    range over the seeds, and the share of seeds whose 90 % interval holds the truth, its Monte
    Carlo standard error beside it:

    | role | search / truth | search held | tv / direct | tv held direct | tv / total | tv held total |
    |---|---|---|---|---|---|---|
    | mediator | 4.62 [2.35, 5.97] | 0.0 | 1.11 [0.55, 2.01] | 0.8 (0.13) | 0.66 [0.33, 1.22] | 0.7 (0.14) |
    | confounder | 1.02 [0.91, 1.86] | 0.9 (0.09) | 1.00 [0.66, 1.36] | 0.9 (0.09) | 0.60 [0.40, 0.80] | 0.1 (0.09) |
    | nested | 1.03 [0.90, 1.87] | 0.9 (0.09) | 1.73 [1.07, 4.26] | 0.1 (0.09) | 1.03 [0.64, 2.56] | 0.7 (0.14) |

    With the queries a control, search's return is read at its truth, its interval holding it at 9
    seeds of 10, and television's at its direct return, as `reading` states. With them left out,
    search is credited with the demand it bought against, 2.4 to 6.0 times its truth, its interval
    above it at every seed, as Chen et al. found. Television's return is then not its total:
    search's spend follows the demand television raised, so search is credited with that part as
    well, and television's reads 0.66 of its total, its interval wide enough to hold both its
    total and its direct return at most seeds. So the mediator's role reads a channel's total only
    where no channel the model reads buys against the proxy's demand; where one does, the nested
    model reads it. Its median is television's total, 1.03 of it, its interval holding it at 7
    seeds of 10; at two it read 2.2 and 2.6 times the total, at one 0.64 of it, and at another,
    whose interval held it, stage one diverged at 41 of its 1000 draws. Stage one's prior on television's return was the default
    moved to the queries' units, `mu` plus the log of the ratio of the queries' mean to the KPI's.
    At the default itself, a return near 1.2 queries a euro against a true 50 or so, stage one read
    almost no path through the queries: television read 1.02 of its direct return and held its
    total at 1 seed of 10. A stage-one prior is in the mediator's units; since ADR 0014's
    amendment of 2026-11-07 the default is moved to them, and the same world holds television's
    total at 7 seeds of 10. The two high readings are stage one's: with no trend, television's
    carryover at a decay near 1 took the queries' drift. With a trend of 3 bends in stage one, the
    study's role `nested, trend`, the total is held at 9 seeds of 10, its median 0.96 of it, 0.64
    to 1.31 over the seeds (ADR 0014, amended 2026-11-07).

- *For the bench's family 2*, a search-demand environment of the same kind is the bench's
  `search-demand` environment (2026-11-07): Track M v2's drawn world, paid shopping's log spend moved by `kappa` times 0.15 per
  stationary standard deviation of the week's sentiment as family 2 moves every channel's, an
  exported control `queries`, the week's sales before media plus `eta` times television's media
  effect, over the base, times a log-normal error of sd 0.02, so it confounds paid shopping and
  mediates television, and a truth that keeps each channel's return as the drawn world states it.

## Alternatives

- *The proposal's control function* (`mu_t = MMM_t + sum_c gamma_c v_ct`): rejected above.
- *The role as a field of `BayesSpec` and `RidgeSpec`*: two specs and the save format would change
  for what one helper states.
- *A default role*: either default misreads half the channels without a word; the role is the
  caller's statement of what each return means.
- *The flag inside `read_panel`*: it would move every panel's issues, a reading nobody asked for.
