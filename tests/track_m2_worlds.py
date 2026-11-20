"""Track M v2's world at a few seeds, kept as a fixture so that the tests reading it run without the
bench (ADR 0074).

The world is causaldyn-bench's `causaldyn_bench.endogenous_mmm`, a clean-room implementation of
Heusch (2026, arXiv:2608.21130): 156 weeks of a retailer's sales whose media spend the business
sets from its sales, over a baseline that carries a sentiment nobody observes. For each kind of
`KINDS` and each seed of `SEEDS`, `tests/fixtures/track_m2_worlds.json` holds what an analyst
observes, the sales, each channel's spend, the promotion and the price as the measurement layer
reads them, and the truth, each channel's return over the history. `reference` is
`EndogenousMediaMix()`, his reference instance (Table II): budget feedback, anticipatory spend,
television's bursts and paid shopping's bidding rule on, true returns near 4.20, 2.90 and 1.56.
`exogenous` is the same generator with the four mechanisms off, so the spend reads nothing of the
sales.

`python tests/track_m2_worlds.py`, the bench on `PYTHONPATH`, writes the fixture again; the
provenance line names the bench's version, its commit and the call. `tests/test_feedback.py`
checks the fixture is what the bench draws, wherever the bench imports.
"""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mixdyn import NATIONAL, Cadence, Channel, MediaPanel, Revenue

FIXTURE = Path(__file__).parent / "fixtures" / "track_m2_worlds.json"
KINDS = {
    "reference": {},
    "exogenous": {
        "budget_feedback": False,
        "anticipatory": False,
        "bursts": False,
        "bidding": False,
    },
}
SEEDS = (0, 1, 2, 3, 4)
BENCH = ("0.1.0", "77c402ad2627913ebd4aec79ef7e9368a3628656")
"""The bench's version and commit that drew the fixture."""
START = np.datetime64("2023-01-02")


@dataclass(frozen=True, eq=False)
class TrackM2World:
    """One history: the sales, each channel's spend `(weeks, channels)`, the promotion and the
    price as read, and each channel's true return over the history."""

    kind: str
    seed: int
    channels: tuple[str, ...]
    sales: np.ndarray
    spend: np.ndarray
    promotion: np.ndarray
    price: np.ndarray
    roas: np.ndarray

    def panel(self) -> MediaPanel:
        """The history as one national geo's weekly panel, the promotion and the price its
        controls, as the scorecard reads an export."""
        weeks = self.sales.size
        return MediaPanel(
            dates=START + np.timedelta64(7, "D") * np.arange(weeks),
            cadence=Cadence.WEEK,
            geos=(NATIONAL,),
            channels=tuple(Channel(c) for c in self.channels),
            controls=("promotion", "price"),
            kpi=self.sales[None],
            kpi_unit=Revenue(),
            spend=self.spend[None],
            currency="EUR",
            control_values=np.column_stack([self.promotion, self.price])[None],
            filled=np.zeros((1, weeks), dtype=bool),
        )


def drawn() -> dict[str, Any]:
    """The fixture's content, drawn from the bench, which must import."""
    from causaldyn_bench.endogenous_mmm import EndogenousMediaMix

    worlds = []
    for kind, switches in KINDS.items():
        for seed in SEEDS:
            w = EndogenousMediaMix(**switches).simulate(seed)
            worlds.append(
                {
                    "kind": kind,
                    "seed": seed,
                    "channels": list(w.channels),
                    "sales": w.sales.tolist(),
                    "spend": w.spend.tolist(),
                    "promotion": w.observed_promotion.astype(float).tolist(),
                    "price": w.observed_price.tolist(),
                    "roas": w.roas().tolist(),
                }
            )
    calls = "; ".join(
        f"{kind}: EndogenousMediaMix({', '.join(f'{k}={v}' for k, v in switches.items())})"
        ".simulate(seed)"
        for kind, switches in KINDS.items()
    )
    version, commit = BENCH
    return {
        "provenance": {
            "generator": f"causaldyn_bench.endogenous_mmm: {calls}",
            "causaldyn-bench": version,
            "commit": commit,
            "numpy": np.__version__,
            "seeds": list(SEEDS),
            "observed": "sales, spend, observed_promotion as 0/1, observed_price",
            "truth": "roas(): each channel's media over its spend, summed over the history",
        },
        "worlds": worlds,
    }


@functools.cache
def worlds() -> tuple[TrackM2World, ...]:
    """Every world the fixture holds."""
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return tuple(
        TrackM2World(
            kind=w["kind"],
            seed=w["seed"],
            channels=tuple(w["channels"]),
            sales=np.array(w["sales"]),
            spend=np.array(w["spend"]),
            promotion=np.array(w["promotion"]),
            price=np.array(w["price"]),
            roas=np.array(w["roas"]),
        )
        for w in data["worlds"]
    )


def world(kind: str, seed: int) -> TrackM2World:
    """The fixture's world of `kind` and `seed`."""
    (found,) = [w for w in worlds() if w.kind == kind and w.seed == seed]
    return found


if __name__ == "__main__":
    FIXTURE.write_text(json.dumps(drawn(), separators=(",", ":")) + "\n", encoding="utf-8")
