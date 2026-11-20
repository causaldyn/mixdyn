# 0026. Each effect's discount, and the ends of a grid

Status: accepted, 2026-10-04

## Context

ADR 0020's pre-registered coverage check failed: on 200 worlds whose channels' effects drift by a
random walk, the 90% projected interval held the last week's coefficient in 0.860 and 0.855 of
them against 0.88, and the effects' best discount was the grid's lowest, 0.97, in 117 of the 200.
Two of its consequences name what this record answers:

- *one discount for every channel's effect*: the effects' family moves together, so two channels
  that drift at different rates share one evolution;
- *a grid stops where its values stop*, and nothing said so: the effects' discount sat at the
  grid's floor in most of those worlds, and on the tests' world the best half-saturation point is
  the grid's highest for both channels. A channel whose best sits at the grid's end has found a
  limit, not a point.

## Decision

- **`PerEffect(values)`.** `DiscountGrid.effects` is `tuple[float, ...] | PerEffect`: a tuple is one
  discount shared by every effect, as before, and `PerEffect` gives each effect its own, every
  combination of `values` over the effects, the last effect's varying fastest. Rejected: a flag
  beside the tuple, two fields whose meaning depends on each other; and values by column, a wider
  surface than any caller needs, since every effect tries the same values.
- **Degrees of freedom.** `DiscountGrid.parameters(spec)` counts one parameter a family the grid
  sets and, with `PerEffect`, one an effect, so it reads the spec; it was a property. The
  likelihood ratio's set widens with them, and the projection over it.
- **The product in `choose_dlm`, pairs in the search.** `choose_dlm` filters the whole product:
  `|values|^k` for `k` effects, times the shared families' values, 192 specs at eight values, two
  effects and three trend discounts. `search_media` moves the discounts in blocks
  (`DiscountGrid.blocks`): every pair of the discounts the grid varies, the trend's, the season's
  and each effect's, at every combination of the pair's values with the rest held, so a pass
  filters `k t |values| + k (k - 1) |values|^2 / 2` specs with `t` trend discounts and no season,
  not the product. Each block is built about the spec the last one moved to. A block-wise
  maximum, as ADR 0020's channel blocks give, and the oracle below checks it against the product.
  - Rejected: one discount at a time. On the tests' world, television's transform held at its
    truth, it stopped 1.24 in log-likelihood below the product's best, where search's and the
    price's discounts trade off and neither moves alone; pairs reach the product's best there and
    in each of six grids and channels probed, of up to 1 536 specs.
  - Rejected: the product in the search. At eight values, four effects and three trend discounts
    it is 12 288 specs a pass, against 480 for the pairs.
- **Ends.** `DlmChoice.edges` names each end of a grid that a kept spec reaches where the
  parameter goes on past it, as an `Edge(parameter, end, best)`: a discount's lowest value and its
  highest where that is below 1, the discount's own bound; a retention's ends but 0 and 1, its
  bounds; and either end of the half-saturation points, as a multiple of the channel's mean
  positive spend. A family or axis of one value has no end, being set and not chosen. `best` says
  whether the likelihood's best spec sits there. `choose_dlm` and `search_media` log the ends as one
  warning with the geo, on their own loggers.
  - *The set, not only the best*: an interval is a union over the set, and a set that reaches an
    end may stop there short of the data's whether or not the best does. A best at an end is the
    stronger sign, and `best` keeps it.
  - Rejected: widening the grid wherever the best sits at an end, until it does not. That
    changes what the likelihood ratio's set is a set of after the data are read, and a grid the
    caller chose is the one a pre-registered check can name.

## Consequences

- *The oracles*, each a test that fails when its implementation is reverted:
  - on a world of three effects, one drifting by a random walk of 0.02 a week, each effect's
    discount chosen on its own puts the drifting one's at 0.97 and the others' at 1, where one
    shared discount sits at 0.99; the set is `chc.dlm.confidence_set`'s at three parameters;
  - a grid's ends are named exactly where a kept spec reaches them: a floor, a ceiling below 1,
    each effect's own with `PerEffect`, never a bound of the parameter, never a family of one
    value;
  - on the tests' world the search names television's half-saturation point at 4 times its mean
    positive spend, the best there, and search's at both ends and the effects' discount at 0.97,
    while a retention of 0 in search's set is not named;
  - with each channel's transform held at its truth, the search over each effect's discount ends
    at the spec that filtering all 128 of the product ranks first, its set a part of the
    product's, and each of its moves raises the likelihood and sets at most two discounts.
- *Mutations*: each of 12 mutants fails a test, among them blocks built once a pass, one discount
  at a time, one degree of freedom for every effect's discount in the search's margin, 1 or a
  retention of 0 named as an end, and a family of one value given ends.
- *Cost.* `choose_dlm`'s product grows as the values to the power of the effects: with many
  effects, a shared tuple, or `search_media`'s blocks.
- *Coverage: the pre-registered check passed.* On 200 worlds of a stream of their own, drawn by ADR
  0020's law with each channel's true transform given to the model, and `choose_dlm` on each
  effect's discount from 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99 and 1 and the trend's from 0.98,
  0.99 and 1, the 90% projected interval at the last week held `beta e^x_T` in 0.960 [0.923, 0.983]
  of the worlds for television and 0.980 [0.950, 0.995] for search (Clopper-Pearson 95%), against
  the same 0.88, set before the run (`evidence/dlm_media_coverage_per_effect.py`, its rule in its
  docstring; its record `evidence/results/dlm_media_coverage_per_effect.json`). The likelihood's
  pick alone held 0.670 and 0.740, and across television's terciles of `sigma_c` the projection held
  1.000, 0.955 and 0.925. On the same worlds ADR 0020's grid held 0.740 and 0.870, and one discount
  shared over this grid's values 0.875 and 0.930; this grid's intervals are a median 1.41 and 1.43
  times as wide as ADR 0020's grid's and 1.23 and 1.21 times the shared one's. The kept set reached
  the effects' floor of 0.80 in 13 worlds for television and 8 for search, and held the truth in 13
  and 7 of them.
- *The arm's grid.* As the check set before it ran, the scorecard's DLM arm takes its grid: each
  effect's discount on its own from 0.80 to 1, a control's as a channel's, and the trend's from
  0.98 to 1.
- A warning per fit can be many in a sweep; it names the geo, and `edges` is on the choice for a
  caller that wants to count them instead.
