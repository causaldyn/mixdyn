"""A geo experiment before it runs: which geos are treated, how alike the two groups are, and the
power of the test the design implies.

Geos are named by their rows: row `g` of the KPI's history and of every feature is geo `g`, as
`mixdyn.data.MediaPanel.kpi` holds them in `panel.geos`' order.

`pair_geos` sorts the geos by a score read over the pre-period, their mean KPI say, and cuts the
order into blocks of adjacent geos, pairs by default. Over one dimension that minimises the summed
range of the blocks, the summed gap within the pairs: two pairs that cross can be uncrossed without
widening either. Where the count does not divide into blocks, the geos left over are those whose
leaving out minimises the same sum, read by dynamic programming over the sorted order, and they are
named. `assign` then treats a fixed share of each block, rounded half up, one geo of a pair, by a
seeded draw: the design randomised within matched pairs of Imai, King and Nall (2009). Each geo is
treated with the same probability within its block, so the difference in means is unbiased over
the draws, and the pairs bound the imbalance any one draw can have, which complete randomisation
does not and a selection of the top geos maximises.

`match_controls` serves a treated set the user fixes: each treated geo's control among the
candidates, one to one, minimising the summed Mahalanobis distance over the pre-period features by
the Hungarian method (`scipy.optimize.linear_sum_assignment`), under the two groups' pooled
within-group covariance. A greedy match in the order of the data can take a control a later treated
geo needed more.

`balance` reads each feature's standardised difference, the treated mean less the control mean over
the pooled standard deviation `sqrt((s_t^2 + s_c^2) / 2)`, and the largest in absolute value.

`power` reads the test's power by simulation from the KPI's own history. Each replication draws a
window among those the history holds, `pre` periods and then `readout`, multiplies the treated geos'
KPI over the readout by `1 + lift` and reads the window. The default reader is the one that reads
the test once it has run, `mixdyn.experiments.read_did` on the groups' summed KPI, its standard
error from walk-forward placebo windows with 13 periods before each, widened to its t at their
effective count (ADR 0055). `log_did` reads the log-ratio gap a period, `g_t`, the treated geos'
mean log KPI less the control geos', each geo weighed alike as a regression of the log KPI on geo
and period effects weighs it: the effect is the mean gap over the readout less its mean over the
pre-period, `log(1 + lift)` in expectation, and its standard error the root mean square of the same
statistic over every placebo window of `readout` periods inside the pre-period, each read against
the rest of the pre-period, widened as `mixdyn.experiments.PlaceboNoise` widens its own. A window
read against the baseline on both its sides is read too tightly where the noise persists, and
`log_did` over-rejects there (ADR 0055); it stays selectable by `read=`. A lift is detected where
the effect passes `z(1 - alpha/2)` standard errors. The power is the share of replications that
detect it; the minimum detectable lift is where the power, interpolated linearly in the lift from 0
and the test's size there, first reaches 80 %. The windows are drawn by one uniform number a
replication, shared by every lift, so the lifts are read on common random numbers, and so are calls
that differ only in the readout or the pre-period at one seed. The history's own noise, its
dependence over time and the geos' levels included, sets the power, so it moves with the readout
and the pre-period, as a formula over the geos' spread across the market cannot.

`audit_power` reads a design's power once over every backtest the history holds rather than by
draws: a backtest is a window of `pre` periods and the `readout` after them, the windows one
period apart, each read once at every lift, at a 5 % level. It refuses a pre-period shorter than
three times the readout, and a history that holds fewer than 8 backtests whose readouts do not
overlap, naming their count and the history it needs: a power read off a few backtests is the
best of few draws, and a design chosen on it promises more than it delivers (mlsynth's
replication of GeoX's design reports a power of 0.77 where 16 backtests promised 0.80). It reports
the backtests' count; its size, the share of them that read a lift where there is none; the
minimum detectable lift at 80 % over them all; and a range, the least and the largest of the same
lift read over the first half of the backtests in time and over the second.

The size is the estimate; whether it differs from 5 % is tested apart. Backtests one period apart
share their readouts' periods, so their rejections are not independent, and a binomial test of
their count flags a reader of the right size too often. The test counts the backtests one
`readout` apart from the first, whose readouts do not overlap, that read a lift where there is
none, and flags the audit where the exact two-sided binomial test of that count against 5 %, twice
the smaller tail, rejects at 1 % (validation/geo_aa_flag.mac). A reader of size 5 % whose verdicts
over disjoint readouts are independent is flagged at most 1 % of the time. The readers' errors
also read the pre-periods, which these backtests still share, so for them the count is not
exactly binomial. Below 104 such backtests no count is too low to flag: the flag reads a size too
large, never one too small. Over 13 it flags a reader of size 0.10 with probability 0.034 and one
of 0.20 with 0.25: it catches a reader far from its size, not one near it.

`stratified_design` is the comparator design, written from the published method alone (Kerman,
Wang and Vaver 2017, "Estimating ad effectiveness using geo experiments in a time-based
regression framework"; the GeoX paper's description of its design, read without its code): the
geos stratified on their pre-period series' mean, coefficient of variation and slope, candidate
designs drawn within the strata, filtered by the time-based regression's fit out of sample and an
A/A check, and ranked by the regression's minimum detectable lift. Each geo's three features are
read over the last `pre` periods: the log of its mean, its standard deviation over its mean, and
its least-squares slope over its mean, each standardised over the geos. The strata are formed
greedily, the two geos closest in the three features first, then the closest two of the rest,
each grown to `size` by the geo nearest its centre; the `geos % size` left over are in neither
group, as `pair_geos` leaves its own. `candidates` designs are drawn by `assign` over the strata,
the distinct ones kept. The pre-period is cut into three, its last `readout` periods the
evaluation, the `readout` before them the validation and the rest the training. A candidate's
regression of its treated geos' summed KPI on its control geos' (`read_tbr`'s) is fitted on the
training and read on the validation, its out-of-sample R^2; refitted on the training and the
validation, its A/A test reads the evaluation's lift against its predictive error under
independent errors, Kerman, Wang and Vaver's, and fails where that lift is past its Student t at
95 %. The candidates at or above the median R^2 that pass the A/A test are ranked by
`(z(0.975) + z(0.8)) se / baseline`, `se` the regression's predictive error over a readout of
`readout` periods fitted on the whole pre-period, the control's sum over it at its mean, and
`baseline` the treated sum's mean over the pre-period times the readout: the least is chosen.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from scipy import optimize, stats

from mixdyn.data import Cadence, Channel, Geo, MediaPanel, Revenue
from mixdyn.decisions.posterior import _whole
from mixdyn.experiments.candidates import _UPPER, _effective_count
from mixdyn.experiments.reading import GeoTest, read_did, read_tbr

if TYPE_CHECKING:
    from mixdyn.calibration import LiftEstimate
    from mixdyn.experiments.reading import GeoReading

type _Array = npt.NDArray[np.float64]

type Reader = Callable[[_Array, _Array, int], tuple[float, float]]
"""A test's reading of its window: the treated and the control geos' KPI, `(geos, periods)`, the
first `pre` periods before the test and the rest its readout, to the effect read and its standard
error, the effect 0 where the test changed nothing."""

_TARGET = 0.8  # the power the minimum detectable lift is read at
_SINGULAR = 1e-10  # a covariance's smallest eigenvalue relative to its largest, below which it is
# read as singular
_WARMUP = 13  # the periods `read_did` and `read_tbr` hold before each placebo window by default
_CHANNEL = Channel("test")
_BACKTESTS = 8  # the fewest backtests whose readouts do not overlap a power audit reads
_LEVEL = 0.05  # the level a power audit reads at
_FLAG_LEVEL = 0.01  # the level of the binomial test that flags a size other than _LEVEL


@dataclass(frozen=True, slots=True)
class Blocks:
    """Geos matched into `blocks`, each a tuple of rows in the order of the score, and the rows
    `left_out` of every block."""

    blocks: tuple[tuple[int, ...], ...]
    left_out: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class GeoDesign:
    """The rows of the `treated` geos and of their `control` geos.

    Raises:
        ValueError: on a group empty, a row not a whole number at least 0, and a row named twice.
    """

    treated: tuple[int, ...]
    control: tuple[int, ...]

    def __post_init__(self) -> None:
        treated, control = tuple(self.treated), tuple(self.control)
        if not treated or not control:
            msg = f"a design needs a treated and a control geo, got {treated} and {control}"
            raise ValueError(msg)
        for row in treated + control:
            _whole(row, "a geo's row", 0)
        if len(set(treated + control)) != len(treated) + len(control):
            msg = f"a geo is named twice in the treated {treated} and the control {control}"
            raise ValueError(msg)
        object.__setattr__(self, "treated", treated)
        object.__setattr__(self, "control", control)


@dataclass(frozen=True, slots=True)
class Balance:
    """Each feature's standardised difference between the treated and the control geos,
    `differences`, and the `largest` in absolute value."""

    differences: Mapping[str, float]
    largest: float


@dataclass(frozen=True, slots=True, eq=False)
class Power:
    """The power of a design's test at each of `lifts`, `power`, its `size`, the share of
    replications that read a lift where there is none, the minimum detectable lift at 80 % `mde`,
    NaN where no lift reaches it, and the count of `windows` the history holds."""

    lifts: _Array
    power: _Array
    size: float
    mde: float
    windows: int


@dataclass(frozen=True, slots=True, eq=False)
class PowerAudit:
    """A design's power over every one of its `backtests`, at a 5 % level: its `power` at each of
    `lifts`; its `size`, the share of the backtests that read a lift where there is none; the
    minimum detectable lift at 80 % over them all, `mde`; and the least and the largest of it read
    over the first half of the backtests in time and over the second, `mde_low` and `mde_high`. A
    lift no backtest reaches is NaN, and so is `mde_high` where a half reaches none. The size is
    tested over the `disjoint` backtests whose readouts do not overlap, of which `rejected` read a
    lift where there is none: `flag_p_value` is the exact two-sided binomial test's p-value against
    a size of 5 %, and the audit is `flagged` where it is at most 0.01."""

    lifts: _Array
    power: _Array
    size: float
    mde: float
    mde_low: float
    mde_high: float
    backtests: int
    disjoint: int
    rejected: int
    flag_p_value: float
    flagged: bool


@dataclass(frozen=True, slots=True)
class StratifiedDesign:
    """The comparator design `stratified_design` chose: its `design`, the `strata` it was drawn
    within, the count of distinct `candidates` drawn and of those `kept` by the R^2 and the A/A
    filters, and the chosen one's out-of-sample `r2` and its time-based regression's minimum
    detectable lift by formula, `mde`, a share of the treated geos' KPI."""

    design: GeoDesign
    strata: Blocks
    candidates: int
    kept: int
    r2: float
    mde: float


def pair_geos(score: npt.ArrayLike, *, size: int = 2) -> Blocks:
    """The geos matched into blocks of `size` adjacent in the order of `score`, one number a geo,
    minimising the summed range of the blocks; the `len(score) % size` geos left out are those that
    minimise it, the later in the order on a tie.

    Raises:
        ValueError: on a score not a one-dimensional finite series, a size not a whole number at
            least 2, and fewer geos than a block's size.
    """
    values = np.array(score, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        msg = f"score must be a one-dimensional finite series, got shape {values.shape}"
        raise ValueError(msg)
    _whole(size, "size", 2)
    n = values.size
    if n < size:
        msg = f"{n} geos hold no block of {size}"
        raise ValueError(msg)
    order = np.argsort(values, kind="stable")
    v = values[order]
    spare = n % size
    cost = np.full((n + 1, spare + 1), math.inf)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(min(i, spare) + 1):
            if (i - j) % size:
                continue
            if i - j >= size:
                cost[i, j] = cost[i - size, j] + v[i - 1] - v[i - size]
            if j and cost[i - 1, j - 1] < cost[i, j]:
                cost[i, j] = cost[i - 1, j - 1]
    blocks: list[tuple[int, ...]] = []
    left: list[int] = []
    i, j = n, spare
    while i:
        if j and cost[i - 1, j - 1] == cost[i, j]:
            left.append(int(order[i - 1]))
            i, j = i - 1, j - 1
        else:
            blocks.append(tuple(int(g) for g in order[i - size : i]))
            i -= size
    return Blocks(tuple(reversed(blocks)), tuple(reversed(left)))


def assign(blocks: Blocks, seed: int, *, share: float = 0.5) -> GeoDesign:
    """A design treating `share` of each block's geos, rounded half up, one of a pair and three of
    a block of five at the default, drawn by `seed`: the same seed draws the same design. The geos
    left out of the blocks are in neither group.

    Raises:
        ValueError: on a seed not a whole number at least 0, and a share that treats none of a
            block or all of it.
    """
    _whole(seed, "seed", 0)
    rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
    treated: list[int] = []
    control: list[int] = []
    for block in blocks.blocks:
        k = math.floor(share * len(block) + 0.5) if math.isfinite(share) else 0
        if not 1 <= k < len(block):
            msg = f"share={share!r} treats {k} of a block of {len(block)}, which leaves no contrast"
            raise ValueError(msg)
        drawn = rng.permutation(len(block))
        treated += [block[i] for i in drawn[:k]]
        control += [block[i] for i in drawn[k:]]
    return GeoDesign(tuple(sorted(treated)), tuple(sorted(control)))


def match_controls(
    treated: Sequence[int], candidates: Sequence[int], features: Mapping[str, npt.ArrayLike]
) -> GeoDesign:
    """Each of `treated`'s control among `candidates`, one to one, minimising the summed
    Mahalanobis distance over `features`, each a value a geo, under the treated and the candidate
    geos' pooled within-group covariance. The design's `control[i]` is `treated[i]`'s. One treated
    geo and one candidate are the only match there is, and no covariance is read.

    Raises:
        ValueError: on no features, a feature not a finite value a geo, fewer candidates than
            treated geos, a geo in both or twice, and, over three geos or more, a pooled
            covariance that is singular, which names the features it cannot tell apart.
    """
    names, x = _features(features)
    GeoDesign(tuple(treated), tuple(candidates))
    if len(candidates) < len(treated):
        msg = f"{len(candidates)} candidates cannot match {len(treated)} treated geos one to one"
        raise ValueError(msg)
    rows = list(treated) + list(candidates)
    if max(rows) >= x.shape[0]:
        msg = f"a geo's row {max(rows)} is past the features' {x.shape[0]} geos"
        raise ValueError(msg)
    if len(rows) == 2:
        return GeoDesign(tuple(treated), tuple(candidates))
    xt, xc = x[list(treated)], x[list(candidates)]
    dof = len(rows) - 2
    if dof < 1:
        msg = "a pooled covariance needs at least three geos"
        raise ValueError(msg)
    centred = np.vstack([xt - xt.mean(axis=0), xc - xc.mean(axis=0)])
    covariance = centred.T @ centred / dof
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if not eigenvalues[0] > _SINGULAR * max(float(eigenvalues[-1]), 0.0):
        null = np.abs(eigenvectors[:, 0])
        tied = [n for n, w in zip(names, null, strict=True) if w > 1e-6 * null.max()]
        msg = (
            f"the pooled covariance of the features {tied} is singular over these geos: they "
            f"do not vary within the groups, or move together, so no distance tells them apart"
        )
        raise ValueError(msg)
    gaps = xt[:, None, :] - xc[None, :, :]
    whitened = gaps @ (eigenvectors / np.sqrt(eigenvalues))
    distance = np.sqrt((whitened**2).sum(axis=-1))
    t, c = optimize.linear_sum_assignment(distance)
    control = np.empty(len(treated), dtype=np.intp)
    control[t] = c
    return GeoDesign(tuple(treated), tuple(int(candidates[int(k)]) for k in control))


def balance(design: GeoDesign, features: Mapping[str, npt.ArrayLike]) -> Balance:
    """Each of `features`' standardised difference over `design`: the treated geos' mean less the
    control geos', over `sqrt((s_t^2 + s_c^2) / 2)`, `s` a group's standard deviation.

    Raises:
        ValueError: on no features, a feature not a finite value a geo, a design's geo past the
            features', a group of one geo, which has no spread, and a feature constant over both
            groups.
    """
    names, x = _features(features)
    rows = design.treated + design.control
    if max(rows) >= x.shape[0]:
        msg = f"a geo's row {max(rows)} is past the features' {x.shape[0]} geos"
        raise ValueError(msg)
    if min(len(design.treated), len(design.control)) < 2:
        msg = "a group of one geo has no spread to standardise a difference by"
        raise ValueError(msg)
    xt, xc = x[list(design.treated)], x[list(design.control)]
    spread = np.sqrt((xt.var(axis=0, ddof=1) + xc.var(axis=0, ddof=1)) / 2.0)
    flat = [n for n, s in zip(names, spread, strict=True) if not s > 0.0]
    if flat:
        msg = f"the features {flat} are constant within both groups: nothing standardises them"
        raise ValueError(msg)
    differences = (xt.mean(axis=0) - xc.mean(axis=0)) / spread
    return Balance(
        MappingProxyType(dict(zip(names, differences.tolist(), strict=True))),
        float(np.abs(differences).max()),
    )


def log_did(treated: _Array, control: _Array, pre: int) -> tuple[float, float]:
    """The pre/post log-ratio difference in differences of a window and its standard error from
    placebo windows of the pre-period; see the module.

    Raises:
        ValueError: on a pre-period that holds fewer than two placebo windows of the readout.
    """
    gap = np.log(treated).mean(axis=0) - np.log(control).mean(axis=0)
    readout = gap.size - pre
    count = pre - readout + 1
    if readout < 1 or count < 2:
        msg = f"a pre-period of {pre} holds {max(count, 0)} placebo windows of {readout}, not 2"
        raise ValueError(msg)
    effect = float(gap[pre:].mean() - gap[:pre].mean())
    sums = np.concatenate([[0.0], np.cumsum(gap[:pre])])
    inside = (sums[readout:] - sums[:count]) / readout
    outside = (sums[pre] - (sums[readout:] - sums[:count])) / (pre - readout)
    placebos = inside - outside
    square = float(placebos @ placebos)
    if not square > 0.0:
        return effect, 0.0
    nu = _effective_count(placebos, square)
    widened = float(stats.t.ppf(_UPPER, nu) / stats.norm.ppf(_UPPER))
    return effect, math.sqrt(square / count) * widened


def _summed(
    reader: Callable[[MediaPanel, GeoTest], tuple[LiftEstimate, GeoReading]],
    warmup: int = _WARMUP,
) -> Reader:
    """`reader`, one of `mixdyn.experiments.reading`'s, as a `Reader`: the window's geos a weekly
    panel, the readout its test, read to the estimate's lift and its standard error widened to a
    t's, so that `power`'s normal test at a 5 % level is the reading's verdict.

    The reader refuses a pre-period that holds fewer than two placebo windows of the readout with
    `warmup` periods before each."""

    def read(treated: _Array, control: _Array, pre: int) -> tuple[float, float]:
        kpi = np.vstack([treated, control])
        geos, periods = kpi.shape
        readout = periods - pre
        count = pre - readout - warmup + 1
        if readout < 1 or count < 2:
            msg = (
                f"a pre-period of {pre} holds {max(count, 0)} placebo windows of {readout} after "
                f"{warmup} periods, not 2"
            )
            raise ValueError(msg)
        names = tuple(Geo(f"g{g}") for g in range(geos))
        dates = np.datetime64("2000-01-03") + np.timedelta64(7, "D") * np.arange(periods)
        panel = MediaPanel(
            dates=dates,
            cadence=Cadence.WEEK,
            geos=names,
            channels=(_CHANNEL,),
            controls=(),
            kpi=kpi,
            kpi_unit=Revenue(),
            spend=np.zeros((geos, periods, 1)),
            currency="EUR",
            control_values=np.zeros((geos, periods, 0)),
            filled=np.zeros((geos, periods), dtype=bool),
        )
        test = GeoTest(
            _CHANNEL,
            names[: treated.shape[0]],
            names[treated.shape[0] :],
            dates[pre],
            dates[-1],
            np.ones(periods),
            np.concatenate([np.ones(pre), np.zeros(readout)]),
            readout,
        )
        estimate, _ = reader(panel, test)
        return estimate.lift, estimate.se

    return read


_read_did = _summed(read_did)
_read_tbr = _summed(read_tbr)


def power(
    kpi_history: npt.ArrayLike,
    design: GeoDesign,
    readout: int,
    lifts: npt.ArrayLike,
    *,
    pre: int,
    reps: int,
    seed: int,
    alpha: float = 0.05,
    read: Reader = _read_did,
) -> Power:
    """The power of `design`'s test of `readout` periods after `pre` at each of `lifts`, by `reps`
    replications drawn from `kpi_history`, `(geos, periods)`, by `seed`, each read by `read`, by
    default `read_did` on the groups' sums; see the module.

    Raises:
        ValueError: on a history not finite and positive or of fewer geos than the design's; a
            readout, a pre-period, a count of replications or a seed not a whole number at least 1,
            2, 1 and 0; no lifts, or a lift not finite or at or below -1, which leaves no KPI; an
            alpha outside `(0, 1)`; a history too short for a window of `pre` and `readout`
            periods; and what `read` refuses, by default a pre-period that holds fewer than two
            placebo windows of the readout with 13 periods before each.
    """
    history = _history(kpi_history, design)
    _whole(readout, "readout", 1)
    _whole(pre, "pre", 2)
    _whole(reps, "reps", 1)
    _whole(seed, "seed", 0)
    grid = _lifts(lifts)
    if not (math.isfinite(alpha) and 0.0 < alpha < 1.0):
        msg = f"alpha={alpha!r} is not in (0, 1)"
        raise ValueError(msg)
    windows = history.shape[1] - pre - readout + 1
    if windows < 1:
        msg = (
            f"a history of {history.shape[1]} periods holds no test of {readout} periods after "
            f"{pre} before it"
        )
        raise ValueError(msg)
    rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
    starts = np.floor(rng.random(reps) * windows).astype(np.intp)
    critical = float(stats.norm.ppf(1.0 - alpha / 2.0))
    treated, control = list(design.treated), list(design.control)
    scaled = np.concatenate([[0.0], grid])
    detected = np.zeros((reps, scaled.size), dtype=bool)
    for r, start in enumerate(starts):
        window = history[:, start : start + pre + readout]
        for k, lift in enumerate(scaled):
            moved = window[treated].copy()
            moved[:, pre:] *= 1.0 + lift
            effect, se = read(moved, window[control], pre)
            detected[r, k] = abs(effect) > critical * se
    rates = detected.mean(axis=0)
    return Power(grid, rates[1:], float(rates[0]), _detectable(scaled, rates), windows)


def audit_power(
    kpi_history: npt.ArrayLike,
    design: GeoDesign,
    readout: int,
    lifts: npt.ArrayLike,
    *,
    pre: int,
    read: Reader = _read_did,
) -> PowerAudit:
    """The power of `design`'s test of `readout` periods after `pre`, at each of `lifts`, read by
    `read` once over every backtest `kpi_history`, `(geos, periods)`, holds, by default with
    `read_did` on the groups' sums, at a 5 % level; see the module.

    Raises:
        ValueError: on a history not finite and positive or of fewer geos than the design's; a
            readout or a pre-period not a whole number at least 1 and 2; a pre-period shorter than
            three times the readout; no lifts, or a lift not finite or at or below -1; a history
            that holds fewer than 8 backtests whose readouts do not overlap; and what `read`
            refuses.
    """
    history = _history(kpi_history, design)
    _whole(readout, "readout", 1)
    _whole(pre, "pre", 2)
    if pre < 3 * readout:
        msg = (
            f"a pre-period of {pre} periods is shorter than three times the test's {readout}: "
            f"it needs {3 * readout}"
        )
        raise ValueError(msg)
    grid = _lifts(lifts)
    backtests = history.shape[1] - pre - readout + 1
    disjoint = max((backtests - 1) // readout + 1, 0)
    if disjoint < _BACKTESTS:
        msg = (
            f"a history of {history.shape[1]} periods holds {disjoint} backtests of a test of "
            f"{readout} periods after {pre} whose readouts do not overlap, fewer than "
            f"{_BACKTESTS}: it needs {pre + _BACKTESTS * readout} periods"
        )
        raise ValueError(msg)
    critical = float(stats.norm.ppf(1.0 - _LEVEL / 2.0))
    treated, control = list(design.treated), list(design.control)
    scaled = np.concatenate([[0.0], grid])
    detected = np.zeros((backtests, scaled.size), dtype=bool)
    for start in range(backtests):
        window = history[:, start : start + pre + readout]
        for k, lift in enumerate(scaled):
            moved = window[treated].copy()
            moved[:, pre:] *= 1.0 + lift
            effect, se = read(moved, window[control], pre)
            detected[start, k] = abs(effect) > critical * se
    rates = detected.mean(axis=0)
    size = float(rates[0])
    half = backtests // 2
    ends = [
        _detectable(scaled, detected[rows].mean(axis=0))
        for rows in (slice(0, half), slice(backtests - half, backtests))
    ]
    finite = [e for e in ends if math.isfinite(e)]
    rejected = int(detected[::readout, 0].sum())
    p_value = _binomial_p_value(rejected, disjoint)
    return PowerAudit(
        grid,
        rates[1:],
        size,
        _detectable(scaled, rates),
        min(finite) if finite else math.nan,
        max(ends) if len(finite) == 2 else math.nan,
        backtests,
        disjoint,
        rejected,
        p_value,
        p_value <= _FLAG_LEVEL,
    )


def _binomial_p_value(rejected: int, count: int) -> float:
    """The exact two-sided binomial test of `rejected` of `count` against a rate of `_LEVEL`: twice
    the smaller tail, at most 1 (validation/geo_aa_flag.mac)."""
    below = float(stats.binom.cdf(rejected, count, _LEVEL))
    above = float(stats.binom.sf(rejected - 1, count, _LEVEL))
    return min(1.0, 2.0 * min(below, above))


def stratified_design(
    kpi_history: npt.ArrayLike,
    readout: int,
    *,
    pre: int,
    seed: int,
    candidates: int = 200,
    size: int = 2,
    share: float = 0.5,
) -> StratifiedDesign:
    """The comparator design over the geos of `kpi_history`, `(geos, periods)`, for a test of
    `readout` periods read after the history's last `pre`: strata of `size` geos alike in their
    series' mean, coefficient of variation and slope, `candidates` designs drawn within them by
    `seed`, each treating `share` of a stratum, filtered by the time-based regression's
    out-of-sample R^2 and an A/A test and the least minimum detectable lift chosen; see the
    module.

    Raises:
        ValueError: on a history not finite and positive; a readout, a pre-period, a count of
            candidates, a size or a seed not a whole number at least 1, 3, 1, 2 and 0; a
            pre-period shorter than three times the readout or longer than the history; fewer
            geos than a stratum's size; what `assign` refuses of the share; and no candidate that
            passes the A/A test.
    """
    history = np.array(kpi_history, dtype=np.float64)
    if history.ndim != 2 or not (np.isfinite(history) & (history > 0.0)).all():
        msg = f"kpi_history must be a finite positive (geos, periods) array, got {history.shape}"
        raise ValueError(msg)
    _whole(readout, "readout", 1)
    _whole(pre, "pre", 3)
    _whole(candidates, "candidates", 1)
    _whole(size, "size", 2)
    _whole(seed, "seed", 0)
    if pre < 3 * readout:
        msg = (
            f"a pre-period of {pre} periods is shorter than three times the test's {readout}: "
            f"it needs {3 * readout}"
        )
        raise ValueError(msg)
    if pre > history.shape[1]:
        msg = f"a pre-period of {pre} periods is longer than the history's {history.shape[1]}"
        raise ValueError(msg)
    if history.shape[0] < size:
        msg = f"{history.shape[0]} geos hold no stratum of {size}"
        raise ValueError(msg)
    series = history[:, -pre:]
    strata = _strata(_standardised(series), size)
    drawn: dict[tuple[int, ...], GeoDesign] = {}
    for child in np.random.SeedSequence(seed).spawn(candidates):
        design = assign(strata, int(child.generate_state(1)[0]), share=share)
        drawn.setdefault(design.treated, design)
    scored = [(design, *_tbr_scores(series, design, readout)) for design in drawn.values()]
    floor = float(np.median([r2 for _, r2, _, _ in scored]))
    kept = [entry for entry in scored if entry[1] >= floor and entry[2]]
    if not kept:
        msg = f"none of the {len(scored)} candidate designs passes the A/A test"
        raise ValueError(msg)
    design, r2, _, mde = min(kept, key=lambda entry: entry[3])
    return StratifiedDesign(design, strata, len(scored), len(kept), r2, mde)


def _standardised(series: _Array) -> _Array:
    """Each geo's log mean, coefficient of variation and least-squares slope over its mean,
    `(geos, 3)`, each standardised over the geos; a feature alike in every geo is 0."""
    mean = series.mean(axis=1)
    t = np.arange(series.shape[1], dtype=np.float64)
    t -= t.mean()
    slope = (series @ t) / float(t @ t)
    features = np.column_stack([np.log(mean), series.std(axis=1, ddof=1) / mean, slope / mean])
    centred = features - features.mean(axis=0)
    spread = centred.std(axis=0)
    return np.divide(centred, spread, out=np.zeros_like(centred), where=spread > 0.0)


def _strata(z: _Array, size: int) -> Blocks:
    """Strata of `size` rows of `z` formed greedily: the closest two rows left, grown by the row
    nearest the stratum's centre; the rows too few for one more stratum are left out."""
    left = list(range(z.shape[0]))
    blocks: list[tuple[int, ...]] = []
    while len(left) >= size:
        points = z[left]
        gaps = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
        gaps[np.diag_indices(len(left))] = np.inf
        i, j = np.unravel_index(int(np.argmin(gaps)), gaps.shape)
        block = [left[i], left[j]]
        while len(block) < size:
            rest = [g for g in left if g not in block]
            centre = z[block].mean(axis=0)
            block.append(rest[int(np.argmin(np.linalg.norm(z[rest] - centre, axis=1)))])
        blocks.append(tuple(sorted(block)))
        left = [g for g in left if g not in block]
    return Blocks(tuple(blocks), tuple(left))


def _tbr_scores(series: _Array, design: GeoDesign, readout: int) -> tuple[float, bool, float]:
    """`design`'s time-based regression over the pre-period `series`: its R^2 on the validation
    out of sample, whether its A/A test on the evaluation passes, and its minimum detectable
    lift by formula; see the module."""
    y = series[list(design.treated)].sum(axis=0)
    x = series[list(design.control)].sum(axis=0)
    n = y.size
    train, fitted = n - 2 * readout, n - readout
    a, b, _ = _ols(x[:train], y[:train])
    held = slice(train, fitted)
    residual = y[held] - a - b * x[held]
    total = y[held] - y[held].mean()
    r2 = 1.0 - float(residual @ residual) / float(total @ total) if total.any() else -math.inf
    a, b, s2 = _ols(x[:fitted], y[:fitted])
    xbar, sxx = x[:fitted].mean(), float(((x[:fitted] - x[:fitted].mean()) ** 2).sum())
    lift = float(np.sum(y[fitted:] - a - b * x[fitted:]))
    ahead = float(np.sum(x[fitted:] - xbar))
    aa_se = math.sqrt(s2 * (readout + readout**2 / fitted + ahead**2 / sxx))
    passes = abs(lift) <= float(stats.t.ppf(_UPPER, fitted - 2)) * aa_se
    _, _, s2 = _ols(x, y)
    se = math.sqrt(s2 * (readout + readout**2 / n))
    z = float(stats.norm.ppf(_UPPER) + stats.norm.ppf(_TARGET))
    return r2, passes, z * se / (readout * float(y.mean()))


def _ols(x: _Array, y: _Array) -> tuple[float, float, float]:
    """The least-squares intercept and slope of `y` on `x` and the residual variance over
    `n - 2` degrees of freedom; a slope of 0 where `x` does not move."""
    dx = x - x.mean()
    spread = float(dx @ dx)
    b = float(dx @ (y - y.mean())) / spread if spread > 0.0 else 0.0
    a = float(y.mean()) - b * float(x.mean())
    residual = y - a - b * x
    return a, b, float(residual @ residual) / max(y.size - 2, 1)


def _detectable(lifts: _Array, rates: _Array) -> float:
    """The least lift at which the power, interpolated linearly from the size at 0 over the
    positive lifts in increasing order, first reaches the target; NaN where none does."""
    positive = np.flatnonzero(lifts > 0.0)
    order = positive[np.argsort(lifts[positive], kind="stable")]
    x = np.concatenate([[0.0], lifts[order]])
    y = np.concatenate([[rates[0]], rates[order]])
    above = np.flatnonzero(y >= _TARGET)
    if not above.size:
        return math.nan
    k = int(above[0])
    if k == 0:
        return 0.0
    return float(x[k - 1] + (x[k] - x[k - 1]) * (_TARGET - y[k - 1]) / (y[k] - y[k - 1]))


def _history(kpi_history: npt.ArrayLike, design: GeoDesign) -> _Array:
    """The history as a finite positive `(geos, periods)` array that holds `design`'s geos."""
    history = np.array(kpi_history, dtype=np.float64)
    if history.ndim != 2 or not (np.isfinite(history) & (history > 0.0)).all():
        msg = f"kpi_history must be a finite positive (geos, periods) array, got {history.shape}"
        raise ValueError(msg)
    if max(design.treated + design.control) >= history.shape[0]:
        msg = f"the design names a geo past the history's {history.shape[0]}"
        raise ValueError(msg)
    return history


def _lifts(lifts: npt.ArrayLike) -> _Array:
    """The lifts as a one-dimensional array, each finite and above -1."""
    grid = np.array(lifts, dtype=np.float64)
    if grid.ndim != 1 or not grid.size or not (np.isfinite(grid) & (grid > -1.0)).all():
        msg = f"lifts must be finite and above -1, at least one, got {grid.tolist()}"
        raise ValueError(msg)
    return grid


def _features(features: Mapping[str, npt.ArrayLike]) -> tuple[list[str], _Array]:
    """The features' names and their values, `(geos, features)`."""
    if not features:
        msg = "no features to read"
        raise ValueError(msg)
    names = list(features)
    columns = [np.array(features[n], dtype=np.float64) for n in names]
    bad = [
        n for n, c in zip(names, columns, strict=True) if c.ndim != 1 or not np.isfinite(c).all()
    ]
    if bad or len({c.size for c in columns}) != 1:
        msg = f"each feature must be a finite value a geo, of one count: {bad or names}"
        raise ValueError(msg)
    return names, np.column_stack(columns)
