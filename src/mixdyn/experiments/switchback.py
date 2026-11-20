"""A switchback test: one national channel switched on and off over blocks of periods, its design
before it runs and its reading after, as a `mixdyn.calibration.LiftEstimate` a model reads.

Where a channel cannot be cut by geography, national TV or a brand sold in one market, the test
randomises time instead (Bojinov, Simchi-Levi and Zhao 2023, "Design and analysis of switchback
experiments", Management Science 69). `switchback_design` cuts the test's periods into blocks of
`block` periods and turns the channel on in exactly `round_half_up(share * blocks)` of them, in an
order drawn from a child of the seed's stream, the same child `assign` draws a geo design from.

A channel's effect carries over: its adstock persists after a switch, so the first periods of a
block after a switch read part of the previous block's state (Hu and Wager 2022, "Switchback
experiments under geometric mixing"). The design names a `washout`, the periods of every block
after a switch left out of the reading; a block in the state of the one before it keeps all its
periods, and the first block, whose state before the test the design does not know, is read as
following a switch. A washout as long as the block leaves nothing and is refused. A test may be
read with a `burn_in` in place of its washout, at least as long: the first `burn_in` periods of
every block after a switch and of the first block left out, of every arrangement the
randomization reading reads, so its test stays exact. It is fixed before the KPI is read: one
chosen after reading it, by the carryover-order test below say, gives the two steps together a
size that is not the test's.

`read_switchback` reads the KPI summed over geos, as the geo readers sum it: the lift is the mean
KPI over the kept periods on less the mean over the kept periods off, times the count of periods
on, the KPI the periods on gained over the channel off. Two methods read its error:

- "randomization", the default: the difference's distribution over the design's own arrangements,
  every way of turning `on_blocks` of the blocks on, each read with the periods its own switches
  keep, enumerated where they are at most 100 000 and otherwise the observed one and 10 000 drawn
  from a second child of the seed's stream; enumerated, it holds a few float matrices of
  arrangements by periods, about 20 bytes an arrangement a period at its peak (318 MiB for 18
  blocks of 20 periods, 48 620 arrangements). The p-value is the share of arrangements whose
  difference is at least as far from 0 as the observed one. The interval inverts the same test
  under a constant effect `tau` a period on: the KPI less `tau` on the periods on, read again
  over every arrangement; each arrangement's difference is linear in `tau`, so the interval's ends
  are found exactly, as the first `tau` on each side of the estimate where the test rejects. It
  holds its size whatever the noise's dependence over time: the KPI is held fixed and only the
  assignment is random.
- "hac": each block's mean over its kept periods, the difference read as a weighted sum of them,
  its error by one of two bandwidths and read as a Student t at `blocks - 2`. "newey-west", the
  default, is Newey and West's (1987): Bartlett weights to lag `floor(4 (blocks / 100)^(2/9))`
  (Newey and West 1994) and a factor `blocks / (blocks - 2)`. "lag-0" reads no lag, the pooled
  two-sample error of the block means, each weighted by its kept periods, exact under a normal
  noise independent from period to period. The blocks' scores are centred on their state's mean, so
  their cross products are negative under such a noise, and the kernel's lags read the
  difference's variance too small: 0.69 to 0.84 of it over 4 to 12 blocks, where lag 0 reads it
  whole. Neither holds its size on every cell of ADR 0056: "newey-west" over-rejects everywhere,
  "lag-0" where a washout meets a noise that persists across blocks. "hac" is a reporting
  fallback, not a test with a size.
- "jackknife": the half-panel jackknife's error. The blocks are cut into two halves in time, the
  first `ceil(blocks / 2)` and the last as many, an odd count's two sharing its middle block, and
  each half's difference is read over its own kept periods, kept as the whole test keeps them.
  The error is half the gap between the two halves' differences, `|d_1 - d_2| / 2`, their spread
  about their mean, read as a Student t at 1 degree of freedom: exact where the halves'
  differences are independent normals of one variance and the whole's is their mean, whatever the
  noise does within a half. Its 95 % quantile is 12.71, so its interval is wide: it reads whether
  the halves agree more than it tests. Its power is a quarter to a half of "randomization"'s on
  ADR 0056's worlds: 0.206 against 0.872 at an effect of 1 a period over 12 blocks of 4 without
  carryover or a washout, 0.108 against 0.387 at retention 0.5, `rho = 0.5` and a washout of 2,
  0.068 against 0.140 at an effect of 0.5. The lift is the other methods', not the jackknife's
  corrected `2 d - (d_1 + d_2) / 2`, which takes out of a carryover's bias only the part that
  falls with the blocks (ADR 0056).

A `carryover` check is reported beside the reading, not used in it: over the blocks after a
switch with at least two kept periods, the first kept period less the mean of the rest, the mean
over blocks on less the mean over blocks off. A washout too short for the channel's carryover
leaves it below 0 for a channel that raises the KPI.

`carryover_order` tests the order a reading assumes by the design's own randomisation: whether the
channel still moves the KPI `order` periods after a switch, `order` the design's washout by
default, the order of Bojinov, Simchi-Levi and Zhao's carryover, an outcome that depends on its
period's assignment and the `order` before it. Its two epochs are, in each block, the period
`order` after the block's start and the periods after it; `g_b` is the first less the mean of the
second, and the statistic the carryover check's at a washout of `order`, the mean of `g_b` over the
blocks on after a switch less the mean over the blocks off after one, 0 where one state has none.
Where the carryover reaches no further than `order` periods and the effect is one value over a
block's periods past them, both epochs read it alike and `g_b` holds no effect in any block,
whichever state the block is in or follows: the statistic over the arrangements the
randomization reading reads, the observed one among them, is then its distribution under that
null, and the p-value, the share of them at least as far from 0 as the observed one, is exact
whatever the noise. A carryover past `order` moves `g_b` in the blocks after a switch, down after
the channel goes on and up after it goes off for a channel that raises the KPI.

The reading's verdict is keyed on the interval's sign, as `GeoReading`'s. The estimate's error is
widened as the geo readers widen theirs, so that its normal 95 % interval holds the reading's 95 %
interval: `se t_df(0.975) / z(0.975)` for "hac", and for "randomization", whose interval need not
be symmetric, the farther of its two ends from the lift over `z(0.975)`. Its readout is the test's
periods, its `planned` spend the spend as run with every period on at the mean spend of the
periods off. Its lift is the steady state's, the effect a period on times the periods on, so it
counts the carry-out of the last blocks on past the test's last period, which a lift over the
readout alone leaves out: by 2.1 % at retention 0.5 and 8.4 % at 0.8 over 12 blocks of 4 (ADR
0056).
"""

from __future__ import annotations

import dataclasses
import itertools
import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy import stats

from mixdyn.calibration import LiftEstimate
from mixdyn.decisions.posterior import _whole
from mixdyn.experiments.candidates import _UPPER

type _Array = npt.NDArray[np.float64]
type Method = Literal["randomization", "hac", "jackknife"]
type Bandwidth = Literal["newey-west", "lag-0"]
type Verdict = Literal["up", "down", "inconclusive"]

_ENUMERATED = 100_000  # the most arrangements the randomization reading enumerates
_DRAWN = 10_000  # the arrangements it draws beside the observed one past that
_TIE = 1e-12  # the relative gap within which two differences read as equally far from 0


@dataclass(frozen=True, slots=True, eq=False)
class SwitchbackDesign:
    """A switchback test over the periods dated `dates`, cut into blocks of `block` periods, the
    channel on in the blocks where `on` is true; the first `washout` periods of every block after
    a switch, and of the first block, are left out of the reading. `share` and `seed` are those it
    was drawn at: `share` turns `round_half_up(share * blocks)` blocks on, as many as `on` does,
    and `seed`'s stream draws the arrangements the randomization reading samples; a design
    written by hand names its own.

    Raises:
        ValueError: on dates that are not increasing dates, a block or a washout not a whole
            number, a washout at or above the block, periods that do not cut into whole blocks,
            fewer than 4 blocks, `on` not one flag a block, no block on or every block on, a share
            outside `(0, 1)` or one that turns another count of blocks on than `on`, and a seed
            not a whole number at least 0.
    """

    dates: npt.NDArray[np.datetime64]
    block: int
    washout: int
    on: tuple[bool, ...]
    share: float
    seed: int

    def __post_init__(self) -> None:
        dates = np.asarray(self.dates)
        if dates.ndim != 1 or not np.issubdtype(dates.dtype, np.datetime64):
            msg = f"the test's periods must be dates, a numpy datetime64 array, got {self.dates!r}"
            raise ValueError(msg)
        dates = dates.astype("datetime64[D]")
        if np.isnat(dates).any() or not (np.diff(dates) > np.timedelta64(0, "D")).all():
            msg = "the test's dates must be increasing"
            raise ValueError(msg)
        _whole(self.block, "block", 1)
        _whole(self.washout, "washout", 0)
        if self.washout >= self.block:
            msg = f"a washout of {self.washout} periods leaves nothing of a block of {self.block}"
            raise ValueError(msg)
        if dates.size % self.block:
            msg = f"{dates.size} periods do not cut into whole blocks of {self.block}"
            raise ValueError(msg)
        blocks = dates.size // self.block
        if blocks < 4:
            msg = f"a switchback test needs at least 4 blocks, got {blocks}"
            raise ValueError(msg)
        on = tuple(self.on)
        if len(on) != blocks or not all(isinstance(b, bool | np.bool_) for b in on):
            msg = f"on must flag each of the {blocks} blocks true or false, got {self.on!r}"
            raise ValueError(msg)
        if not 1 <= sum(on) < blocks:
            msg = f"{sum(on)} of {blocks} blocks on leaves no contrast"
            raise ValueError(msg)
        if not (math.isfinite(self.share) and 0.0 < self.share < 1.0):
            msg = f"share={self.share!r} is not a share in (0, 1)"
            raise ValueError(msg)
        if math.floor(self.share * blocks + 0.5) != sum(on):
            msg = (
                f"share={self.share!r} turns {math.floor(self.share * blocks + 0.5)} of {blocks} "
                f"blocks on, not the {sum(on)} on flags"
            )
            raise ValueError(msg)
        _whole(self.seed, "seed", 0)
        dates.setflags(write=False)
        object.__setattr__(self, "dates", dates)
        object.__setattr__(self, "on", tuple(bool(b) for b in on))

    @property
    def periods(self) -> int:
        return int(self.dates.size)

    @property
    def blocks(self) -> int:
        return len(self.on)

    @property
    def on_blocks(self) -> int:
        return sum(self.on)

    @property
    def assignment(self) -> npt.NDArray[np.bool_]:
        """Each period's state, true where the channel is on."""
        return np.repeat(np.array(self.on), self.block)

    @property
    def kept(self) -> npt.NDArray[np.bool_]:
        """Each period's place in the reading, false in a washout."""
        return _kept(np.array([self.on]), self.block, self.washout)[0]


@dataclass(frozen=True, slots=True)
class SwitchbackReading:
    """A switchback test's reading by `method`: the `lift` over the periods on and its standard
    error `se`, for "randomization" the standard deviation of the lift over the arrangements, for
    "hac" Newey and West's and for "jackknife" half the gap between its halves' lifts, read as a
    Student t of `df` degrees of freedom (infinite for "randomization", which reads no t); the
    interval `[lower, upper]` at `level`; the `p_value` against no lift; the `verdict`, "up" when
    the interval lies above 0, "down" when below and "inconclusive" when it holds 0; the count of
    `arrangements` read, 0 for "hac" and "jackknife", and whether they were all `enumerated`; and
    the `carryover` check, NaN where no block after a switch keeps two periods in each state.
    """

    method: Method
    lift: float
    se: float
    df: float
    level: float
    lower: float
    upper: float
    p_value: float
    verdict: Verdict
    arrangements: int
    enumerated: bool
    carryover: float


@dataclass(frozen=True, slots=True)
class CarryoverOrder:
    """A switchback test's two-epoch test of its carryover's order: whether the channel still
    moves the KPI `order` periods after a switch. `statistic` is the first epoch less the second
    over the blocks on after a switch, less the same over the blocks off after one, and `p_value`
    the share of the `arrangements` read whose statistic is at least as far from 0, all of the
    design's where `enumerated`. A small p-value says the carryover reaches past `order`, and a
    washout or a burn-in of `order` periods leaves part of it in the reading.
    """

    order: int
    statistic: float
    p_value: float
    arrangements: int
    enumerated: bool


def switchback_design(
    periods: npt.ArrayLike, block: int, washout: int, *, seed: int, share: float = 0.5
) -> SwitchbackDesign:
    """A switchback test over the periods dated `periods`, in blocks of `block` periods, the
    channel on in `round_half_up(share * blocks)` of them, in an order drawn by `seed` from
    `numpy.random.SeedSequence(seed).spawn(1)[0]`: the same seed draws the same design. The first
    `washout` periods of every block after a switch are left out of the reading.

    Raises:
        ValueError: on what `SwitchbackDesign` refuses, and a share that turns no block on or
            every block on.
    """
    _whole(seed, "seed", 0)
    _whole(block, "block", 1)
    if not (math.isfinite(share) and 0.0 < share < 1.0):
        msg = f"share={share!r} is not a share in (0, 1)"
        raise ValueError(msg)
    dates = np.asarray(periods)
    blocks = dates.size // block if dates.ndim == 1 else 0
    on_blocks = math.floor(share * blocks + 0.5)
    if blocks >= 4 and not 1 <= on_blocks < blocks:
        msg = f"share={share!r} turns {on_blocks} of {blocks} blocks on, which leaves no contrast"
        raise ValueError(msg)
    rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
    on = np.zeros(blocks, dtype=bool)
    on[rng.permutation(blocks)[:on_blocks]] = True
    return SwitchbackDesign(dates, block, washout, tuple(bool(b) for b in on), share, seed)


def read_switchback(
    kpi: npt.ArrayLike,
    spend: npt.ArrayLike,
    design: SwitchbackDesign,
    *,
    level: float = 0.95,
    method: Method = "randomization",
    bandwidth: Bandwidth = "newey-west",
    burn_in: int | None = None,
) -> tuple[LiftEstimate, SwitchbackReading]:
    """`design`'s test read from `kpi` and the channel's `spend` as run, each a value a period of
    the design or a row a geo of them, summed over the geos; see the module.

    `bandwidth` names the "hac" error's, "newey-west" or "lag-0"; the other methods read none.
    `burn_in` leaves out the first `burn_in` periods of every block after a switch in place of the
    design's washout; `None`, the default, reads the washout.

    Raises:
        ValueError: on `level` outside `(0, 1)`; `method` none of the readings; `bandwidth` neither
            bandwidth; a burn-in not a whole number, shorter than the washout or at or above the
            block; a KPI or a spend not the design's periods long or not finite; a KPI whose kept
            periods are all alike; for "randomization", arrangements too few for the test to
            reject at `level` or at 95 %, which leave its interval unbounded; for "hac", block
            means each at their state's mean; for "jackknife", a half that keeps no period on or
            none off, and two halves that read the same difference; and what `LiftEstimate`
            refuses of the spend, a spend the same on and off among them.
    """
    if not (math.isfinite(level) and 0.0 < level < 1.0):
        msg = f"level={level!r} is not a level in (0, 1)"
        raise ValueError(msg)
    if method not in ("randomization", "hac", "jackknife"):
        msg = f"method must be 'randomization', 'hac' or 'jackknife', got {method!r}"
        raise ValueError(msg)
    if bandwidth not in ("newey-west", "lag-0"):
        msg = f"bandwidth must be 'newey-west' or 'lag-0', got {bandwidth!r}"
        raise ValueError(msg)
    if burn_in is not None:
        _whole(burn_in, "burn_in", 0)
        if burn_in < design.washout:
            msg = (
                f"a burn-in of {burn_in} periods reads periods the design's washout of "
                f"{design.washout} leaves out"
            )
            raise ValueError(msg)
        if burn_in >= design.block:
            msg = f"a burn-in of {burn_in} periods leaves nothing of a block of {design.block}"
            raise ValueError(msg)
        design = dataclasses.replace(design, washout=burn_in)
    y = _summed(kpi, design.periods, "KPI")
    paid = _summed(spend, design.periods, "spend")
    on, kept = design.assignment, design.kept
    if np.ptp(y[kept]) == 0.0:
        msg = "the KPI is the same in every kept period: nothing reads its noise"
        raise ValueError(msg)
    count = design.block * design.on_blocks
    observed = float(y[on & kept].mean() - y[~on & kept].mean())
    lift = count * observed
    alpha = 1.0 - level
    if method == "randomization":
        p_value, spread, reaches, arrangements, enumerated = _randomization(
            y, design, observed, (alpha, 2.0 * (1.0 - _UPPER))
        )
        (below, above), (below95, above95) = reaches
        if not all(math.isfinite(u) for u in (below, above, below95, above95)):
            msg = (
                f"{arrangements} arrangements of {design.blocks} blocks are too few for the test "
                f"to reject at a level of {level} or of 0.95, the estimate's: its interval is "
                "unbounded"
            )
            raise ValueError(msg)
        se, df = count * spread, math.inf
        lower, upper = lift - count * below, lift + count * above
        widened = count * max(below95, above95) / float(stats.norm.ppf(_UPPER))
    else:
        if method == "hac":
            se = count * _newey_west(y, design, bandwidth)
            if not se > 0.0:
                msg = "every block's mean is its state's: nothing reads the blocks' noise"
                raise ValueError(msg)
            df = float(design.blocks - 2)
        else:
            first, second = _halves(y, design)
            se = count * abs(first - second) / 2.0
            if not se > 0.0:
                msg = "the test's two halves read the same difference: nothing reads its spread"
                raise ValueError(msg)
            df = 1.0
        t = stats.t(df)
        half = float(t.ppf(1.0 - alpha / 2.0)) * se
        lower, upper = lift - half, lift + half
        p_value = 2.0 * float(t.sf(abs(lift) / se))
        widened = se * float(t.ppf(_UPPER) / stats.norm.ppf(_UPPER))
        arrangements, enumerated = 0, False
    verdict: Verdict = "up" if lower > 0.0 else "down" if upper < 0.0 else "inconclusive"
    reading = SwitchbackReading(
        method,
        lift,
        se,
        df,
        level,
        lower,
        upper,
        p_value,
        verdict,
        arrangements,
        enumerated,
        _carryover(y, design),
    )
    planned = paid.copy()
    planned[on] = paid[~on].mean()
    estimate = LiftEstimate(paid, planned, design.periods, lift, widened, end=design.dates[-1])
    return estimate, reading


def carryover_order(
    kpi: npt.ArrayLike, design: SwitchbackDesign, *, order: int | None = None
) -> CarryoverOrder:
    """The two-epoch test that `design`'s channel moves `kpi` no further than `order` periods after
    a switch, `order` the design's washout by default, `kpi` a value a period of the design or a
    row a geo of them, summed over the geos; see the module.

    Its power reaches 0.8 only where the carryover past `order` is large or the blocks many. Read
    at order 1 on ADR 0056's world (blocks of 4, a washout of 1, an AR(1) noise of variance 1 at
    `rho`), with a carryover of order 2 that reaches the second period after a switch at `c` noise
    standard deviations, over 1000 worlds a cell: over 24 blocks it reaches 0.8 at `c = 1.33` for
    `rho` 0, 0.5 and 0.8 (0.914 to 1.000), and at `c = 1` where `rho` is 0.5 or 0.8 (0.843, 0.997);
    over 12 blocks only at `c = 1.33` and `rho = 0.8` (0.926), and at `c = 1.33` and `rho = 0` it
    is 0.500. A test of few blocks that does not reject has not shown the carryover short.

    Raises:
        ValueError: on an order not a whole number at least 0, or one that leaves a block's second
            epoch empty, at `block - 1` periods or more; and a KPI not the design's periods long or
            not finite.
    """
    order = design.washout if order is None else order
    _whole(order, "order", 0)
    if order + 2 > design.block:
        msg = (
            f"a block of {design.block} periods holds no second epoch after an order of {order}: "
            f"the test needs blocks of {order + 2}"
        )
        raise ValueError(msg)
    y = _summed(kpi, design.periods, "KPI").reshape(design.blocks, design.block)
    gaps = y[:, order] - y[:, order + 1 :].mean(axis=1)
    rows, enumerated = _arrangements(design)
    statistics = _epochs(gaps, rows)
    observed = float(_epochs(gaps, np.array([design.on]))[0])
    p_value = float(np.mean(np.abs(statistics) >= abs(observed) * (1.0 - _TIE)))
    return CarryoverOrder(order, observed, p_value, int(rows.shape[0]), enumerated)


def _epochs(gaps: _Array, rows: npt.NDArray[np.bool_]) -> _Array:
    """For each arrangement of `rows`, the mean of `gaps` over its blocks on after a switch less
    the mean over its blocks off after one, the first block among neither; 0 where one state has
    no such block."""
    later = rows[:, 1:]
    switched = later != rows[:, :-1]
    up, down = (switched & later).astype(np.float64), (switched & ~later).astype(np.float64)
    n_up, n_down = up.sum(axis=1), down.sum(axis=1)
    both = (n_up > 0.0) & (n_down > 0.0)
    rest = gaps[1:]
    return np.where(
        both, up @ rest / np.maximum(n_up, 1.0) - down @ rest / np.maximum(n_down, 1.0), 0.0
    )


def _halves(y: _Array, design: SwitchbackDesign) -> tuple[float, float]:
    """Each half's difference of its kept periods' means on and off, the first `ceil(blocks / 2)`
    blocks and the last as many, each period kept as the whole test keeps it."""
    span = (design.blocks + 1) // 2 * design.block
    on, kept = design.assignment, design.kept
    differences = []
    for name, half in (("first", slice(0, span)), ("second", slice(design.periods - span, None))):
        up, down = (on & kept)[half], (~on & kept)[half]
        if not up.any() or not down.any():
            msg = (
                f"the jackknife reads each half of the test on its own, and the {name} half keeps "
                f"no period {'off' if up.any() else 'on'}"
            )
            raise ValueError(msg)
        differences.append(float(y[half][up].mean() - y[half][down].mean()))
    return differences[0], differences[1]


def _summed(values: npt.ArrayLike, periods: int, name: str) -> _Array:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim == 2:
        x = x.sum(axis=0)
    if x.ndim != 1 or x.size != periods:
        msg = f"the {name} must run the design's {periods} periods, a row a geo, got {x.shape}"
        raise ValueError(msg)
    if not np.isfinite(x).all():
        msg = f"the {name} must be finite in every period of the test"
        raise ValueError(msg)
    return x


def _kept(on: npt.NDArray[np.bool_], block: int, washout: int) -> npt.NDArray[np.bool_]:
    """For each arrangement, a row of `on` a flag a block, each period's place in the reading:
    the first `washout` periods of the first block and of every block after a switch are out."""
    switched = np.ones(on.shape, dtype=bool)
    switched[:, 1:] = on[:, 1:] != on[:, :-1]
    kept = np.ones((on.shape[0], on.shape[1], block), dtype=bool)
    kept[:, :, :washout] = ~switched[:, :, None]
    return kept.reshape(on.shape[0], -1)


def _arrangements(design: SwitchbackDesign) -> tuple[npt.NDArray[np.bool_], bool]:
    """The arrangements the randomization reading reads, a row each, the observed one among them,
    and whether they are all of them."""
    blocks, k = design.blocks, design.on_blocks
    if math.comb(blocks, k) <= _ENUMERATED:
        rows = np.zeros((math.comb(blocks, k), blocks), dtype=bool)
        for i, chosen in enumerate(itertools.combinations(range(blocks), k)):
            rows[i, list(chosen)] = True
        return rows, True
    rng = np.random.default_rng(np.random.SeedSequence(design.seed).spawn(2)[1])
    order = np.argsort(rng.random((_DRAWN, blocks)), axis=1)
    rows = np.zeros((_DRAWN + 1, blocks), dtype=bool)
    rows[0] = design.on
    np.put_along_axis(rows[1:], order[:, :k], values=True, axis=1)
    return rows, False


def _rerandomised(
    y: _Array, rows: npt.NDArray[np.bool_], design: SwitchbackDesign
) -> tuple[_Array, _Array]:
    """Under each arrangement of `rows`, the difference of `y`'s means over its kept periods on
    and off, and the same difference of the observed assignment's flags, the slope of the first
    in a constant effect a period on taken out of `y`."""
    state = np.repeat(rows, design.block, axis=1)
    kept = _kept(rows, design.block, design.washout)
    up, down = (state & kept).astype(np.float64), (~state & kept).astype(np.float64)
    n_up, n_down = up.sum(axis=1), down.sum(axis=1)
    flags = design.assignment.astype(np.float64)
    differences = up @ y / n_up - down @ y / n_down
    slopes = up @ flags / n_up - down @ flags / n_down
    return differences, slopes


def _randomization(
    y: _Array, design: SwitchbackDesign, observed: float, alphas: tuple[float, ...]
) -> tuple[float, float, list[tuple[float, float]], int, bool]:
    """The randomization test of the difference `observed`: its p-value against no effect, the
    difference's standard deviation over the arrangements, at each of `alphas` how far below and
    above `observed` its interval reaches, the count of arrangements and whether they are all."""
    rows, enumerated = _arrangements(design)
    differences, slopes = _rerandomised(y, rows, design)
    # the observed arrangement, and one whose flags are the observed ones reversed, is as far from
    # 0 as itself at every effect: rounding must not move it off its tie. Their slopes are ratios
    # of counts, exactly 1 and -1; their differences less the observed one's need not be 0
    shifted = differences - observed * slopes
    shifted[np.abs(shifted) <= _TIE * float(np.abs(differences).max())] = 0.0
    p_value = float(np.mean(np.abs(differences) >= abs(observed) * (1.0 - _TIE)))
    reaches = [(_reach(shifted, -slopes, a), _reach(shifted, slopes, a)) for a in alphas]
    return p_value, float(differences.std()), reaches, int(rows.shape[0]), enumerated


def _reach(a: _Array, c: _Array, alpha: float) -> float:
    """The least `u > 0` past which the randomization test rejects at `alpha`, `inf` where it
    never does: arrangement `z` is as far from 0 as the observed one while `|a_z - c_z u| >= u`,
    and the test rejects where at most `alpha` of them are. Each arrangement fails on one interval
    of `u > 0`, between the roots of `(c^2 - 1) u^2 - 2 a c u + a^2`, which is at least 0 at 0."""
    q2, q1, q0 = c * c - 1.0, -2.0 * a * c, a * a
    starts = np.full(a.size, np.inf)
    ends = np.full(a.size, np.inf)
    linear = q2 == 0.0
    falls = linear & (q1 < 0.0)
    starts[falls] = q0[falls] / -q1[falls]
    disc = q1 * q1 - 4.0 * q2 * q0
    real = ~linear & (disc > 0.0)
    root = np.sqrt(np.where(real, disc, 0.0))
    # stable roots: q = -(q1 + sign(q1) root) / 2, the roots q / q2 and q0 / q
    q = -0.5 * (q1 + np.where(q1 < 0.0, -root, root))
    with np.errstate(divide="ignore", invalid="ignore"):
        one, two = q / q2, np.where(q != 0.0, q0 / q, 0.0)
    small, large = np.minimum(one, two), np.maximum(one, two)
    between = real & (q2 > 0.0) & (small > 0.0)
    starts[between], ends[between] = small[between], large[between]
    beyond = real & (q2 < 0.0)
    starts[beyond] = np.maximum(large[beyond], 0.0)
    starts[~linear & (q2 < 0.0) & (q0 == 0.0)] = 0.0
    need = a.size * (1.0 - alpha) - 1e-9
    points = np.sort(starts[np.isfinite(starts)])
    if points.size == 0:
        return math.inf
    opened = np.searchsorted(np.sort(starts), points, side="right")
    closed = np.searchsorted(np.sort(ends), points, side="right")
    failing = np.flatnonzero(opened - closed >= need)
    return float(points[failing[0]]) if failing.size else math.inf


def _newey_west(y: _Array, design: SwitchbackDesign, bandwidth: Bandwidth) -> float:
    """The difference's standard error over the blocks' means of their kept periods: Newey and
    West's, or at "lag-0" the pooled two-sample one."""
    kept = design.kept.reshape(design.blocks, design.block)
    sums = (y.reshape(design.blocks, design.block) * kept).sum(axis=1)
    sizes = kept.sum(axis=1).astype(np.float64)
    means = sums / sizes
    on = np.array(design.on)
    weights = np.where(on, sizes / sizes[on].sum(), -sizes / sizes[~on].sum())
    centre = np.where(on, sums[on].sum() / sizes[on].sum(), sums[~on].sum() / sizes[~on].sum())
    blocks = design.blocks
    if bandwidth == "lag-0":
        pooled = float(sizes @ (means - centre) ** 2) / (blocks - 2)
        return math.sqrt(pooled * (1.0 / sizes[on].sum() + 1.0 / sizes[~on].sum()))
    scores = weights * (means - centre)
    lags = math.floor(4.0 * (blocks / 100.0) ** (2.0 / 9.0))
    variance = float(scores @ scores)
    for h in range(1, lags + 1):
        variance += 2.0 * (1.0 - h / (lags + 1)) * float(scores[h:] @ scores[:-h])
    return math.sqrt(max(variance, 0.0) * blocks / (blocks - 2))


def _carryover(y: _Array, design: SwitchbackDesign) -> float:
    """The first kept period of each block after a switch less the mean of its other kept
    periods, the mean over blocks on less the mean over blocks off; NaN without a block of each."""
    first: dict[bool, list[float]] = {True: [], False: []}
    kept = design.kept
    for b in range(1, design.blocks):
        if design.on[b] == design.on[b - 1]:
            continue
        span = slice(b * design.block, (b + 1) * design.block)
        values = y[span][kept[span]]
        if values.size >= 2:
            first[design.on[b]].append(float(values[0] - values[1:].mean()))
    if not first[True] or not first[False]:
        return math.nan
    return float(np.mean(first[True]) - np.mean(first[False]))
