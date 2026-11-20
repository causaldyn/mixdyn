"""MediaPanel holds its invariants however it is built, not only when `read_panel` builds it."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from mixdyn import Cadence, Channel, Geo, MediaPanel, Revenue


def build(**overrides: Any) -> MediaPanel:
    fields: dict[str, Any] = {
        "dates": np.datetime64("2024-01-01") + np.timedelta64(7, "D") * np.arange(3),
        "cadence": Cadence.WEEK,
        "geos": (Geo("x"),),
        "channels": (Channel("tv"),),
        "controls": (),
        "kpi": np.array([[1.0, 2.0, 3.0]]),
        "kpi_unit": Revenue(),
        "spend": np.ones((1, 3, 1)),
        "currency": "USD",
        "control_values": np.zeros((1, 3, 0)),
        "filled": np.zeros((1, 3), dtype=bool),
    }
    fields.update(overrides)
    return MediaPanel(**fields)


def test_a_valid_panel_builds() -> None:
    assert build().kpi.flags.writeable is False


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        (
            {"dates": np.array(["2024-01-01", "2024-01-08", "2024-01-22"], "datetime64[D]")},
            "not one regular week calendar",
        ),
        (
            {
                "dates": np.array(["2024-01-15", "2024-02-15", "2024-03-15"], "datetime64[D]"),
                "cadence": Cadence.MONTH,
            },
            "not one regular month calendar",
        ),
        ({"spend": -np.ones((1, 3, 1))}, "finite and non-negative"),
        ({"spend": np.ones((1, 2, 1))}, r"spend has shape \(1, 2, 1\), expected \(1, 3, 1\)"),
        ({"kpi": np.array([[1.0, np.inf, 3.0]])}, "kpi is infinite"),
        ({"filled": np.array([[False, True, False]])}, "a filled step must have NaN kpi"),
        ({"geos": (Geo("x"), Geo("x"))}, "geo names repeat"),
        ({"currency": "dollars"}, "ISO 4217"),
        ({"hierarchy": {Geo("x"): Geo("x")}}, "cycle"),
    ],
)
def test_an_invalid_panel_is_refused(overrides: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        build(**overrides)
