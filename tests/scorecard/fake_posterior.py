"""A stand-in for the static arm's two fits: synthetic draws on an export's channels, so that an
arm built on that fit runs a world in seconds."""

import math

import numpy as np
from chc import response

from mixdyn import Channel
from mixdyn.decisions import Readings
from scorecard.arms import bayes, voi

CHAINS, DRAWS, HELD = 2, 100, 40


class Fake:
    """Two fits' synthetic draws on an export's channels, from the export's seed and its rows, the
    tighter the more rows it reads, `draws` a chain; each call's rows kept in `calls`."""

    def __init__(self, draws: int = DRAWS) -> None:
        self.draws = draws
        self.calls: list[tuple[tuple[str, int], ...]] = []

    def __call__(self, observation, sampler) -> voi.Posterior:
        rows = tuple(zip(observation.lift.channel, observation.lift.start, strict=True))
        self.calls.append(rows)
        spend = np.asarray(observation.spend, dtype=float)
        mean = spend.mean(axis=0)
        rng = np.random.default_rng([observation.seed, len(rows)])
        n, c = 2 * CHAINS * self.draws, len(observation.channels)
        spread = 0.6 / (1.0 + len(rows)) + 0.1
        every = Readings(
            channels=tuple(Channel(x) for x in observation.channels),
            retention=rng.uniform(0.3, 0.6, (n, c)),
            half_saturation=mean * rng.uniform(0.8, 1.2, (n, c)),
            slope=np.full((n, c), np.nan),
            coefficient=2.0 * mean * np.exp(rng.normal(0.3, spread, (n, c))),
            tanh=np.ones(n, dtype=bool),
            fit=np.repeat([0, 1], CHAINS * self.draws),
            chain=np.tile(np.repeat(np.arange(CHAINS), self.draws), 2),
            stacking=np.array([0.6, 0.4]),
            pointwise=None,
            length=observation.kernel_length,
        )
        held = np.linspace(0, n - 1, HELD).round().astype(int)
        drawn = tuple(
            tuple(
                response.Channel(
                    response.GeometricAdstock(
                        float(every.retention[s, j]), length=every.length, normalized=True
                    ),
                    response.Tanh(float(every.half_saturation[s, j]) / math.atanh(0.5)),
                    float(every.coefficient[s, j]),
                )
                for j in range(c)
            )
            for s in held
        )
        roi = every.coefficient[held] * 0.5 / mean
        flags = {"hill weight": 0.0, "hill min ess bulk": 100.0}
        fitted = bayes.Fitted(every, drawn, roi, None, flags, {}, {"forecast": "a stand-in"}, 100.0)
        return voi.Posterior(every, fitted)
