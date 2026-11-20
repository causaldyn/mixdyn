"""A demand proxy, query volume for paid search, and the one role a model gives it (ADR 0074).

Paid search buys against demand: its ad shows where someone searches, so its spend rises with the
queries, and the queries move the KPI on their own. Read without them, search is credited with the
demand it bought against; Chen et al. (2018, "Bias Correction For Paid Search In Media Mix
Modeling", Google) found naive search returns 3 to 15 times what geo experiments read. For search
the queries are a confounder, and the model reads them as a control. For the other media they are
a mediator: television raises the searches for the brand, and a model that holds the queries fixed
reads television's effect beside them, its effect through the demand it raised left out. One model
cannot hold them both ways, as Meridian's documentation on Google query volume says, so
`DemandProxy.role` names the one this model holds, and has no default.

`DemandProxy.controls` is what a model reads of the proxy, for `RidgeSpec`'s and `BayesSpec`'s
controls; `DemandProxy.reading` what a channel's return is under the role:

- `"adjusted"`: a channel the proxy confounds, in the confounder's role: its return with the
  demand it bought against held;
- `"direct"`: any other channel in the confounder's role: its return beside the demand it raises;
- `"confounded"`: a channel the proxy confounds, in the mediator's role: its return with the
  demand it bought against, too high where its spend rises with demand;
- `"total"`: any other channel in the mediator's role: its return with the demand it raises.

The proxy holds the confounding as far as it measures the demand: a query count that misses part
of it, or a demand that moves the KPI not in proportion to it, leaves the rest in search's return.

A `"total"` return is the channel's total only where no channel the model reads buys against the
demand: where search does, search is credited with the demand television raised as with the rest,
and television's return reads below its total. The nested model (`mixdyn.models.nested`) reads
both returns: stage one the proxy on the other media, its prior on each return stated in the
proxy's units, and stage two the KPI with the proxy a control, the confounder's model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mixdyn.models.ridge_mmm import Control

type Role = Literal["confounder", "mediator"]
"""What a demand proxy is to the model: a control, `"confounder"`, or left out, `"mediator"`."""

type Reading = Literal["adjusted", "direct", "confounded", "total"]
"""What a channel's return is under a demand proxy's role; see the module."""

_ROLES = ("confounder", "mediator")


@dataclass(frozen=True, slots=True)
class DemandProxy:
    """The panel's control `column`, a demand proxy such as query volume, in the one `role` this
    model gives it; `searched` names the channels whose spend follows the demand it measures.

    Raises:
        ValueError: on a role not one of `"confounder"` and `"mediator"`, no channel searched, or
            the proxy's column among them.
    """

    column: str
    role: Role
    searched: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.role not in _ROLES:
            msg = f"a demand proxy's role is one of {_ROLES}, got {self.role!r}"
            raise ValueError(msg)
        if not self.searched or self.column in self.searched:
            msg = (
                f"a demand proxy confounds at least one channel and is not one, got {self.column!r}"
                f" for {list(self.searched)}"
            )
            raise ValueError(msg)

    def controls(self) -> tuple[Control, ...]:
        """What the model reads of the proxy: a control held at or above 0, demand raising the
        KPI, in the confounder's role; nothing in the mediator's."""
        return (Control(self.column, lower=0.0),) if self.role == "confounder" else ()

    def reading(self, channel: str) -> Reading:
        """What `channel`'s return is under the role; see the module."""
        searched = channel in self.searched
        if self.role == "confounder":
            return "adjusted" if searched else "direct"
        return "confounded" if searched else "total"


__all__ = ["DemandProxy", "Reading", "Role"]
