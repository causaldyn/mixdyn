"""Composite procedures: a procedure that plays, world by world, one of two arms' plans, the choice
read off one arm's record, so that it is scored and claimed without a fit of its own.

A composite names its `router`, the arm whose record says whether a world is routed (its flag
`ROUTED`), and the arm a routed world plays, `routed`; every other world plays the router's own
plan. The composite's score on a world is the played arm's score as the bench scored that arm's
record: its regret, its failure, its claim and its record, with `composite` beside them, which names
the arm played, and its cost, the router's every world and the routed arm's where it is played, as
the procedure would spend them. A router whose fit failed routes nothing: the composite fails with
it and plays the status quo, as the bench scores every failed fit. A world that does not hold both
arms holds no composite.

`gs`, the tested-rung rule GS(1.645) (ADR 0088): the `anchored` arm reads the world's lift tests
and its kernel and routes a world to `stacked`, the engine's static Bayesian plan, where the tests
read a convex curve or the kernel runs past the tests' readout; elsewhere it plays its own plan.
`gs curved`, the tested-rung rule GS-ML(0.7) (ADR 0089), is `gs` with the arm's variant
`anchored curved` in the arm's place: the same routes, read off that variant's record, and the
variant's plan, water-filled along the curves its tests read, everywhere else. `gs held`
(ADR 0093) is `gs curved` with the variant `anchored held` in that variant's place: `gs curved`
world for world where no channel's weekly spend follows the week's number, and where one does,
that variant's own plan, the channel at its status quo, and no route: `stacked`'s plan holds no
channel, so that variant's record says the world is not routed.

A composite is assembled wherever the scorecard has joined a world's arms: a run with its joined
root (`scorecard.claim.read_run`), and with the roots beside it (`scorecard.claim.joined_runs`), and
a pilot's arms with the stored run's (`scorecard.development.joined`). Its score is recomputed from
its two arms' each time, so assembling it twice gives it once.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    from causaldyn_bench.scorecard.scoring import WorldRecord

ROUTED = "routed"
"""The router's flag: true where the world plays the routed arm."""
SECONDS = ("cpu_seconds", "wall_seconds")
"""The costs a composite sums over the arms it ran, where every one of them states it."""


@dataclass(frozen=True, slots=True)
class Composite:
    """A composite procedure: its `name`, the arm whose record routes each world, `router`, and
    the arm a routed world plays, `routed`."""

    name: str
    router: str
    routed: str


GS = Composite("gs", router="anchored", routed="stacked")
"""The tested-rung rule GS(1.645) (ADR 0088)."""
GS_CURVED = Composite("gs curved", router="anchored curved", routed="stacked")
"""The tested-rung rule GS-ML(0.7) (ADR 0089)."""
GS_HELD = Composite("gs held", router="anchored held", routed="stacked")
"""`GS_CURVED` under the time guard (ADR 0093)."""
COMPOSITES = (GS, GS_CURVED, GS_HELD)
"""The composites the scorecard assembles."""


def parts(name: str, composites: Sequence[Composite] = COMPOSITES) -> tuple[str, ...]:
    """The arms whose records hold the plans the procedure `name` plays: a composite's router and
    the arm its routed worlds play; any other procedure, itself."""
    for composite in composites:
        if composite.name == name:
            return (composite.router, composite.routed)
    return (name,)


def compose(record: WorldRecord, composites: Sequence[Composite] = COMPOSITES) -> WorldRecord:
    """`record` with each of `composites` that both of its arms are scored in, see the module."""
    arms = dict(record.arms)
    for composite in composites:
        if composite.router not in arms or composite.routed not in arms:
            continue
        router = arms[composite.router]
        routed = router.failure is None and bool(router.record["flags"][ROUTED])
        name = composite.routed if routed else composite.router
        played = arms[name]
        ran = [router, played] if routed else [router]
        costs = [s.record.get("cost") or {} for s in ran]
        cost = {k: sum(float(c[k]) for c in costs) for k in SECONDS if all(k in c for c in costs)}
        held: dict[str, Any] = {"router": composite.router, "played": name, "routed": routed}
        arms[composite.name] = dataclasses.replace(
            played, record=played.record | {"cost": cost, "composite": held}
        )
    return dataclasses.replace(record, arms=arms)
