# 0004. The engines share a decomposition and a curve per channel, not a base class

Status: accepted, 2026-10-02

## Context

A `Model` protocol was planned with the first engine and deferred twice: a protocol drawn from one
engine fixes that engine's shape on the others. There are now two engines and three fits:
`DlmFit` and `GeoDlmFit` of the dynamic linear model, `RidgeFit` of the ridge model (ADR 0008).

What they share, by name and meaning: `dates`, the fit's window of the panel's calendar; `y`, the
KPI over it; `estimated`, the model's value at each period; and `contributions`, each term's part
of it. The dynamic models also give `base` and `seasonality`; the ridge model's trend and season
are contributions under their terms' names.

What differs is most of the rest:
- the dynamic model's coefficients move and its forecast takes a plan and gives a predictive
  distribution; the ridge model's are static and its forecast is the design times them;
- a geo DLM fits every geo at once, the ridge model one geo;
- the ridge model's hyperparameters are searched as part of fitting (`search_ridge`), the dynamic
  model's chosen by a step of their own (`choose_dlm`).

One thing reads across engines today: the decisions. `plan_geos` reads a `chc.response.Channel`
for each geo's channel, which a ridge fit hands over (`RidgeFit.channels()`) and a dynamic fit does
not yet.

## Decision

- **No base class and no `Model` protocol.** An engine is functions that return its own frozen
  fit, each with the parameters its model has.
- **Two contracts hold across engines, by name.**
  - *The decomposition:* every fit's `dates`, `y`, `estimated` and `contributions`, arrays over its
    window with the meanings above.
  - *The decisions' input:* a `chc.response.Channel` for each channel, carryover and curve, which
    is what `chc.allocation` plans on.
- **A `typing.Protocol` for the decomposition is drawn when the first reader across engines lands**
  (the scorecard harness or an application's decomposition), from these fields, and ty then checks
  every fit against it. A fit that cannot meet it then is a finding about the protocol, not a reason to
  bend the fit.

## Consequences

- A new engine adds functions and a fit type, and nothing to subclass.
- Until the protocol is drawn, a rename of a shared field in one fit is caught by review, not by
  the type checker.
- `RidgeFit.channels()` keys its channels by name, and `plan_geos` keys each geo's by the panel's
  `Channel`, a `NewType` of `str`; the reader that joins them wraps the names.

## Alternatives considered

- **A `Model` base class with `fit`, `predict` and `plan`**, in scikit-learn's style. Rejected: the
  dynamic model's forecast takes a plan and returns a distribution and the ridge model's a point;
  `fit(X, y)` hides the calendar and the raw panel both engines read; a geo DLM fits all geos at
  once.
- **The protocol now, from the four fits.** Rejected: nothing would read it, so the checker would
  hold a shape no caller relies on, and the first reader's needs, intervals, draws or geos, are not
  known yet.
