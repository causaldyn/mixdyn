# 0029. A plan for a posterior's worst share, where no lift test has run

Status: accepted, 2026-10-05; amended 2026-10-06: on CHC 0.13.0 `cvar_allocate` reads each gain
on the curves and searches S-shaped curves, so the plan never does worse than the reference on the
curves either (Consequences)

## Context

Most panels have no lift test, and a posterior fitted without one is wide. ADR 0019 planned a
posterior's expected return and left CHC's `cvar_allocate` out: it weighs every reading alike, and
it was not in the CHC release the engine pinned. A pre-registered pilot on Track M v2's worlds
without tests then planned the stacked Hill and `tanh` fits' 400 draws by `cvar_allocate` against
the status quo at levels 1, 0.5, 0.2 and 0.1. Its rule kept an arm on an environment unless the
arm's paired 95 % interval of regret per euro less the status quo's lay above 0, and chose the
least mean regret on the drawn worlds among the arms kept on both. Every level was kept, and 0.1
had the least. The scorecard's Bayesian arm plans so (ADR 0021), from readings it builds itself;
the engine had no such plan. CHC 0.11 brought `cvar_allocate` within the engine's pin.

## Decision

- **`plan_cvar(panel, geo, readings, budget, periods, limits, *, against, level=0.1,
  draws=400)`**: the split within the ranges that spends the budget for the most mean gain over
  `against` in the worst `level` share of `draws` of the readings' draws, the gain's conditional
  value at risk, by `cvar_allocate` on each draw's channels as `chc.response` reads them, the
  geo's history carried in. `CvarPlan` holds the spend and the reference a period, the budget, the
  level, each draw's gain over the reference, the CVaR and the bound no split's CVaR passes. The
  reference gains 0 under every draw, so the plan never does worse in that share than keeping it.
- **The draws weigh alike**, as `cvar_allocate` reads them. Each fit gives its weight's share of
  `draws` by largest remainders, an earlier fit first on a tie, evenly spaced over its own draws as
  `BayesFit.channel_draws` spaces a fit's; a fit holding fewer draws than its share is refused.
  For two fits that is the arm's rule, the Hill's weight times 400 rounded, except on an exact
  half, where Python's `round` takes the even count. Largest remainders extend it to any number of
  fits, and no share falls below 0.
- **`against` has no default.** It must spend the budget, and the plan in place is the user's to
  name: the current spend `PosteriorPlan` reports, the mean over the last `periods` periods,
  spends it only where the budget was set from it, and the bench's status quo is the last year's
  mean.
- **`level` defaults to 0.1**, the pilot's choice, and **`draws` to 400**, the arm's.
- **The arithmetic is CHC's**, in JAX's precision; ADR 0019's planner stays NumPy's.

## Consequences

- *The oracle* is ADR 0019's three worlds: on each, the plan for the worst tenth against the
  recorded status quo is `cvar_allocate`'s on the readings the arm builds, through
  `BayesFit.channel_draws`, its spend, gains, CVaR and bound bit for bit. On the reference world the
  Hill's weight, 3e-10, gives it no draw.
- *What the tests hold besides*: over the 400 draws read by default, the CVaR is the mean of the
  worst 80 gains at 0.2 and of all 400 at 1, to a relative `1e-12`, and lies between 0 and the
  bound; the shares by largest remainders on three cases worked by hand, a tie and a fit of
  weight 0 among them; and the refusals.
- *Mutations*: 16 of 16 fail a test, among them each step of the shares, each part of a draw's
  channel, the defaults, the history and the grid's check that `plan_posterior` and `worths` now
  make outside `_inputs`.
- The scorecard's arm keeps its own readings for the drift family's run, whose pilot runs on it.
  Moving it onto `plan_cvar` changes no plan but on an exact half, and waits until that run is
  scored.
- *On CHC 0.13.0* (amended 2026-10-06). Before it, `cvar_allocate` read a split's return and the
  reference's on the curves' envelopes from zero spend, so on S-shaped curves a split could report
  a gain in the worst share and lose to the reference on the curves. 0.13.0 reads each gain on the
  curves, keeps the envelopes for the bound, and searches S-shaped curves in boxes to the best
  split; the engine's floor is 0.13.0. The drift family's pilot, planned again from the draws each
  record kept with no refit: the scorecard's `cvar 0.1` split lost to the status quo in the worst
  tenth of its own readings in 2 of 120 worlds on 0.12.1, by as much as 48.3, and in none on
  0.13.0, which moved 52 splits and raised the worst tenth's mean gain by 29.2 on average. The
  scorecard's CVaR arms run on 0.13.0 at least.

## Alternatives considered

- **Stacking's weights as draws repeated**: rejected. The weights are rarely rational, and the
  readings would grow many times over for every plan.
- **Weights in `cvar_allocate` itself**: the Rockafellar–Uryasev objective takes them directly.
  Left as a proposal for CHC: the engine would wait for a release, and the pilot planned shares
  weighed alike.
- **The current spend as the default reference**: rejected, for the reason above. A default that
  is refused for most budgets is a default only for the budget it was set from.
- **`plan_posterior`'s grid with a CVaR objective**: the CVaR of a split's gain is not a sum over
  channels, so the channels' envelopes, which make the grid planner exact, do not apply.
