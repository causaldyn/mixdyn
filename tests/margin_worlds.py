"""The margin rule's worlds (`fixtures/margin_worlds.json`): six of M2's scored worlds at k = 0 in
the development set's fit split, one each of families 0, 2, 3, 5, 7 and 4, every array the rule
reads of each world's export, and what the research code that found the rule made of it; and one
world of family 14, whose kernel is as long as its history, on which that code raised. Beside them
(`fixtures/margin_noise_worlds.json`), what the research code of the noise mixture made of the
seven worlds: the curved regression's margins at the status quo and their covariance, the error
models' posterior means, the columns its design kept, the shrunk means and the rule's own plan."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

PATH = Path(__file__).parent / "fixtures" / "margin_worlds.json"
WORLDS: list[dict[str, Any]] = json.loads(PATH.read_text())["worlds"]
PARITY = [w for w in WORLDS if "error" not in w["research"]]
"""The worlds the research code fitted."""
LONG = next(w for w in WORLDS if w["family"] == 14)
"""Family 14's world: a kernel of the history's 156 weeks."""
NOISE: dict[int, dict[str, Any]] = {
    w["seed"]: w
    for w in json.loads(PATH.with_name("margin_noise_worlds.json").read_text())["worlds"]
}
"""The noise mixture's research readings of each world, by its seed."""


def name(world: dict[str, Any]) -> str:
    return f"family-{world['family']}-{world['environment']}-{world['seed']}"


def arrays(world: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """What `fit_margins` reads of the world: sales, spend, the promotion, the price, the kernel."""
    return (
        np.asarray(world["sales"], dtype=float),
        np.asarray(world["spend"], dtype=float),
        np.asarray(world["controls"]["promotion"], dtype=float),
        np.asarray(world["controls"]["price"], dtype=float),
        int(world["kernel_length"]),
    )
