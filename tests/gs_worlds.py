"""The tested-rung rule's worlds (`fixtures/gs_worlds.json`): eight of M2's scored worlds at the
tested rung in the development set's fit split, of families 0, 2, 3, 5, 7 and 4, routed by the rule
and not, every array the rule reads of each world's export, its lift rows whole; what the research
code that found the rule made of it; and M2's recorded plan of `stacked` for it, with its regret.
Beside them (`fixtures/gs_curved_worlds.json`), what the research code of the curved rule made of
the same worlds: each channel's posterior weight on the line, the rule's own plan and its regret."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PATH = Path(__file__).parent / "fixtures" / "gs_worlds.json"
WORLDS: list[dict[str, Any]] = json.loads(PATH.read_text())["worlds"]
ROUTED = [w for w in WORLDS if w["research"]["routed"]]
"""The worlds the rule plays `stacked` in."""
CURVED: dict[int, dict[str, Any]] = {
    w["seed"]: w for w in json.loads(PATH.with_name("gs_curved_worlds.json").read_text())["worlds"]
}
"""The curved rule's research readings of each world, by its seed."""


def name(world: dict[str, Any]) -> str:
    routed = "routed" if world["research"]["routed"] else "kept"
    return f"family-{world['family']}-{world['environment']}-{world['seed']}-{routed}"
