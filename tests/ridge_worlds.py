"""One geo's panel, a world of three channels whose transforms are known, and geo tests of it, for
the ridge engine's tests."""

from __future__ import annotations

import numpy as np
from chc.lift import GeoArm, LiftTest

from mixdyn import Cadence, Channel, Geo, MediaPanel, Revenue
from mixdyn.models.ridge_mmm import Media
from mixdyn.transforms import Geometric, Hill, WeibullCdf, carry_over, saturate

NATIONAL = Geo("national")
PRE = 4
"""The weeks each of `world_and_tests`'s tests reads before its treated group's spend changes."""


def panel_of(
    kpi: np.ndarray,
    spend: np.ndarray,
    controls: dict[str, np.ndarray] | None = None,
    start: str = "2024-01-01",
) -> MediaPanel:
    controls = controls or {}
    weeks = kpi.size
    return MediaPanel(
        dates=np.datetime64(start) + np.timedelta64(7, "D") * np.arange(weeks),
        cadence=Cadence.WEEK,
        geos=(NATIONAL,),
        channels=tuple(Channel(f"c{j}") for j in range(spend.shape[1])),
        controls=tuple(controls),
        kpi=kpi[None, :],
        kpi_unit=Revenue(),
        spend=spend[None, :, :],
        currency="EUR",
        control_values=np.stack(list(controls.values()) or [np.empty(weeks)], axis=-1)[
            None, :, : len(controls)
        ],
        filled=np.zeros((1, weeks), dtype=bool),
    )


def world(seed: int = 5, weeks: int = 120) -> tuple[np.ndarray, list[Media], np.ndarray]:
    rng = np.random.default_rng(seed)
    spend = rng.gamma(2.0, 50.0, (weeks, 3)) * (rng.random((weeks, 3)) < 0.8)
    media = [
        Media("c0", Geometric(0.9), Hill(2.0, 0.5)),
        Media("c1", WeibullCdf(1.5, 3.0), Hill(1.0, 0.3)),
        Media("c2", Geometric(0.2), Hill(3.0, 0.8)),
    ]
    return spend, media, np.array([400.0, 250.0, 120.0])


def saturated(media: Media, half: float, spend: np.ndarray) -> np.ndarray:
    return saturate(carry_over(spend, media.adstock), media.hill, half)


def world_and_tests(
    test_noise: float,
) -> tuple[MediaPanel, list[Media], np.ndarray, dict[str, list[LiftTest]], dict[str, list]]:
    """A world's KPI, one week of it missing, and its geo tests, with what each test reads: its gap
    in saturated spend and its observed gap, both at the market's scale. c0 goes dark in a group of
    0.3 of the market against one of 0.6, and doubles market-wide; c2 goes dark; c1 is untested.
    Each test runs 22 weeks of the market's planned spend, then reads `PRE` more before the treated
    group's spend changes and 8 after."""
    spend, media, beta = world()
    rng = np.random.default_rng(11)
    halves = [m.hill.gamma * carry_over(spend[:, j], m.adstock).max() for j, m in enumerate(media)]
    columns = np.column_stack([saturated(m, halves[j], spend[:, j]) for j, m in enumerate(media)])
    kpi = 1000.0 + columns @ beta + rng.normal(0.0, 30.0, 120)
    kpi[60] = np.nan
    tests: dict[str, list[LiftTest]] = {}
    reads: dict[str, list] = {}
    for j, start, factor, shares in [
        (0, 40, 0.0, (0.3, 0.6)),
        (0, 70, 2.0, (1.0, 1.0)),
        (2, 50, 0.0, (1.0, 1.0)),
    ]:
        m = media[j]
        planned = spend[start : start + 30 + PRE, j]
        treated = np.concatenate([planned[: 22 + PRE], factor * planned[22 + PRE :]])
        paths = (treated, planned)
        effects = [beta[j] * saturated(m, halves[j], path)[22:] for path in paths]
        base = 2000.0 + 100.0 * np.sin(np.arange(8.0 + PRE))
        test = LiftTest(
            *(
                GeoArm(
                    share * path,
                    share * (base + effect) + rng.normal(0.0, test_noise, 8 + PRE),
                    share,
                )
                for share, path, effect in zip(shares, paths, effects, strict=True)
            )
        )
        gap = saturated(m, halves[j], treated)[22:] - saturated(m, halves[j], planned)[22:]
        tests.setdefault(m.channel, []).append(test)
        reads.setdefault(m.channel, []).append((gap, test.difference))
    return panel_of(kpi, spend), media, beta, tests, reads
