# 0016. One definition of each return

Status: accepted, 2026-10-03; amended 2026-10-28 by ADR 0053: `RoiPrior(counts="carried")` states
the one-geo model's prior on this ADR's `roi` with its carry-out counted; the default still counts
the window, and the models over geos and the nested model refuse it; amended 2026-11-06 by ADR
0070: a conversions KPI's cost per incremental KPI and its marginal cost are `roi` and
`marginal_roi` in the KPI's units, inverted, and a cost with no value carries its reason.

## Context

Every engine hands its channels to the planner as `chc.response.Channel`s, a kernel, a curve and a
coefficient, but nothing turned a channel into the numbers a report shows. A return can be read
many ways, and a report can show several at once: a coefficient reported as a return, an
elasticity reported as a return, a contribution over the spend's stock in place of the spend, a
net return beside a gross marginal one, a marginal return that drops the carryover. Each reading moves a decision: a channel whose margin seems to beat its average gets
more budget.

## Decision

- **`mixdyn.decisions.returns(channel, spend, window, *, revenue_per_kpi)`**, one definition of
  each, all read off `chc.response`'s functions, the window's spend the spend as paid:
  - `contribution`, the channel's revenue in the window's periods, carried over from all of its
    spend before them: what a decomposition attributes to it there;
  - `roi`, the revenue the window's spend earns per currency unit of it, **all of its carryover
    counted**: the series runs on past its end for the kernel's length with nothing spent;
  - `marginal_roi`, the revenue of one more currency unit spread over the window as its spend is,
    carryover counted as in `roi`;
  - `steady_state_marginal_roi`, the revenue of one more currency unit a period once the window's
    mean spend a period has been held for longer than the kernel remembers;
  - `carryover`, the share of a unit's adstock that lands after the period it is spent in,
    `1 - w_0 / sum(w)`, which on a linear curve is the share of its return.
- **Returns are gross**: a return of 1 pays the spend back, the net return is `roi - 1`, and a
  return pays back in profit at `1 / margin`. No function reports a net return beside a gross one.
- **`portfolio_roi`**: several channels' revenue over their spend, each channel at its own revenue
  per KPI.

## Consequences

- *Returns on worlds whose answer is known* (`tests/test_returns.py`). 100 a week for 52 weeks
  into `y = 1000 + 2 adstock(x, r)` returns 4 and 10 a unit at retentions of 0.5 and 0.8, on the
  5 200 spent, where the adstock it builds sums to 10 200 and 24 000. On a Hill the coefficient,
  2.5, is 850 times the marginal return. Held at 100 a week behind a retention of 0.8, a unit's
  marginal return is 0.00139, where differencing the curve at one period's spend reads 0.00250 and
  drops the carryover; the marginal at 100 over the one at 1 is 3.06 %. A linear channel's return
  and marginal return are both 2. A window's interval holds its point and the world's return,
  where the last week's coefficient interval would hold neither. Six channels each returning 1
  return 1, and two at 2 and 1.5 on 1 000 and 3 000 of spend return 1.625. At a retention of 0.5
  the carryover share is 0.5. The posterior on the decay narrows to under a sixth of its prior's
  spread when the KPI carries the channel through 0.6, and keeps over half of it when it does not.
- *Mutations*: each of 20 mutants of the module fails a test that fits no posterior: a return
  without the carryover past the series' end, the series' spend or mean read for the window's, a
  return in KPI units, the carryover share inverted, each refusal dropped, and a portfolio
  weighted otherwise than by spend. Three survived the first tests, the steady state at the
  series' mean, the share inverted, which a retention of 0.5 cannot show, and an infinite spend
  accepted; the tests that fail them were added.
- A Bayesian fit's returns are read at each draw: `channel_draws(count)` of the one-geo, the
  geo and the nested fit hands each channel as `chc.response` reads it at `count` draws, evenly
  spaced over the chains, and a return's posterior is `returns` at each, its interval their
  quantiles. At every draw the channels contribute the draw's return parameter times the window's
  spend, the identity each model is parametrised by, to `1e-12`; on a world whose KPI carries the
  channel through a retention of 0.6 into a Hill, the 90 % intervals of the window's return and of
  its marginal return hold the world's. The planner reads the channels at the posterior medians.
- The models' `roi` parameter, the window's contribution over its spend, on which ADR 0010's prior
  is stated, is not `roi` here: it drops the carryover past the window's end, by `r^(T - t)` of
  each unit spent at `t`, 7.7 % of a constant spend's over 52 weeks at a retention of 0.8.

## Alternatives considered

- **The window's contribution over its spend as the return**, Meridian's ROI and the models'
  parameter. Rejected as the reported return: a window that ends at the panel's last period loses
  the carryover it caused, more the later its spend and the longer the kernel, so two windows'
  returns are not comparable. It remains the contribution, stated as such.
- **A net return**, what an ROI report often shows. Rejected as the default: a planner compares a
  return with 1 or with `1 / margin`, and a net return beside a gross marginal one reads a linear
  channel as if its margin beat its average.
- **A marginal return at a single period's spend.** Rejected: it drops the carryover,
  which at a retention of 0.8 is four fifths of what a unit earns.
