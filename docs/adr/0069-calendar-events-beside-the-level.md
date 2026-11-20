# 0069. Calendar events beside the level

Status: accepted, 2026-11-06; adds `MediaPanel.events`, which the DLM, the ridge and the one-geo
Bayesian model read as regressors, and changes no fit of a panel without them

## Context

ADR 0002 left calendar events out of the panel until an engine read them. Until now a holiday
reached a model through the level, the season or a control column the caller built by hand. The
level is a local level: it follows a lasting change and leaves a one-week spike in the noise. A
season of a few harmonics cannot hold a week. Planners spend into their events: more media in
Black Friday's week, at Christmas, around Easter. A model that has no term for the event then
credits the channel with the event's lift, and its return is high by the lift over the extra
spend.

## Decision

- **The panel holds events by their days.** `MediaPanel.events` maps a name to its days, each
  inside the calendar, from its first label to the day before the period after its last. A name
  is its own, no channel's or control's. `read_panel(..., events=...)` reads them and reports a day
  off the calendar or a name taken as a `BAD_SPEC` error.
- **An event is a column of the periods its days fall in.** `MediaPanel.event_column(name)` is 1
  in each period one of its days falls in, the last period labelled on or before the day, and 0
  elsewhere; two days in one period count once. A recurring holiday is one event with a day a
  year, and so one coefficient. An event is the same in every geo; a regional one is a control.
- **Models read an event by its name, as they read a control.** `MediaPanel.regressors(names,
  geo, periods)` reads controls and events alike.
  - The DLM: `Effect(name, 1.0)`, a static coefficient beside the level, the season and the
    channels; a transform on an event is refused, as on a control.
  - The ridge and the one-geo Bayesian model: `Control(name)`, a column of the design, its sign
    held as a control's may be.
  - The models over geos do not read events yet: a name that is not a control is refused there.
- **A panel without events reads as before.** `regressors` indexes the controls as the engines
  did, so the columns' memory order, and the bits of every mean and spread over them, are
  unchanged.

## Consequences

- *The measurement* (`scripts/study_events.py`, worlds 0 to 199). 156 weeks of a KPI with a yearly
  cycle of two harmonics, one channel returning 0.8 a currency unit with no carryover or curve, and
  three events a year, Easter Sunday, Black Friday and Christmas Day, lifting the KPI by 250, 400
  and 300 in their weeks, where the channel's spend rises from 100 to 250 a week. Each world is
  fitted by the DLM, a level of discount 0.98 and a static season, the channel's coefficient
  static, with and without the three events as static effects. The channel's return over the 156
  weeks is read off 1 000 draws of the decomposition.

  | model | 90 % holds 0.8 | 95 % holds 0.8 | mean 90 % interval |
  |---|---|---|---|
  | the level and the season alone | 0 of 200 | 0 of 200 | [2.231, 2.544] |
  | with the three events | 181 of 200, 0.905 | 190 of 200, 0.950 | [0.612, 0.977] |

  The Monte Carlo standard error of a coverage of 0.9 over 200 worlds is 0.021. The level alone
  reads the channel's return as about three times the world's: the events' lift over the spend
  they drew.
- *The tests* (`tests/test_events.py`): days read into the periods they fall in at a daily, a
  weekly and a monthly cadence, a calendar's last day kept and the next refused; names taken and
  days off the calendar refused, by the panel and by `read_panel`; controls and events read
  together in a geo's periods; the ridge recovering a sale's lift of 300 and a fair's of 150 to
  `1e-3` on a world with no noise, the Bayesian model's standardised column the event's; and on
  worlds 0 to 39 of the measurement, the DLM's interval from its last state holds the return in 30
  or more with the events and in none without.
- *A panel without events* reads as before: the hashes ADR 0068 lists are unchanged by this change.
- *Mutations*: a day read into the period after it, the calendar's end read inclusive, and the
  DLM's event columns dropped each fail a test.
- Forecasts read an event's planned values as a control's: a forecast past the calendar is given
  each event's column over its horizon.

## Alternatives considered

- **A control column per event, built by the caller.** It works, and it is what the engines read
  underneath. Rejected as the interface: every caller re-derives which period a day falls in, a
  weekly panel's week from a Thursday, and a recurring holiday becomes a column per year unless
  the caller merges them.
- **Prophet's holiday windows**, days before and after an event with a coefficient each. Not now:
  a weekly panel's window is the week, and a window is an event of more days, which the mapping
  already holds.
- **Events as a block of the DLM's state with their own discount.** Rejected: an event recurs once
  a year, so a moving coefficient has one observation per move; a static coefficient per event is
  the model the data can read.
