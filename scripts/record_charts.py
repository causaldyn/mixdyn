"""Draw the README's and the benchmark page's charts as SVG, light and dark, from the runs' records
and the README's own example.

    uv run python scripts/record_charts.py

writes `docs/assets/{closest-rival,every-comparison,first-plan}-{light,dark}.svg`:

- **closest-rival**: for each family and rung of the record, mixdyn's arm less the rival whose
  95 % interval reaches highest, the comparison nearest to breaking the claim, with its interval
  and Hall's reader's word;
- **every-comparison**: every comparison of the record, one dot a family-rung, by rival;
- **first-plan**: the README's first example, its fitted margins against the truth it was drawn
  with and the plan it makes.

The record is the 37 family-rungs the claim holds and family 4 at k = 0, which no claim holds: M4's
and M5's national verdicts and M3's geo ones, as `scripts/readme_record.py` reads them. Each file is
SVG text written from those numbers alone, with no plotting library, so the same records write the
same bytes: `tests/test_readme.py` fails where a committed file is not what this script writes.
"""

from __future__ import annotations

import contextlib
import io
import math
import re
import sys
from collections.abc import Callable, Generator
from dataclasses import dataclass
from typing import Any

import numpy as np

import readme_record as record

ASSETS = record.ROOT / "docs" / "assets"
DELTA = 0.05
MINUS = record.MINUS
APOSTROPHE, TYPOGRAPHIC = "'", "\u2019"  # a chart's text sets its apostrophes typographically
TIMES = "\u00d7"
FONT = "system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
WORDS = ("beats", "ties", "loses to")


@dataclass(frozen=True, slots=True)
class Theme:
    """The chart's tokens in one colour scheme: surfaces, inks, rules, and the diverging pair the
    reader's words take (mixdyn beats the rival, ties, loses to it) with its grey midpoint."""

    name: str
    surface: str
    ink: str
    ink2: str
    muted: str
    grid: str
    axis: str
    ring: str
    beats: str
    ties: str
    loses: str
    wash: float
    band: float

    def word(self, word: str) -> str:
        return {"beats": self.beats, "ties": self.ties, "loses to": self.loses}[word]


# The validated reference palette's tokens: blue and orange are its diverging poles, at each mode's
# own step, and the grey is its muted ink.
LIGHT = Theme(
    "light",
    "#fcfcfb",
    "#0b0b0b",
    "#52514e",
    "#898781",
    "#e1e0d9",
    "#c3c2b7",
    "#0b0b0b",
    "#2a78d6",
    "#898781",
    "#eb6834",
    0.08,
    0.14,
)
DARK = Theme(
    "dark",
    "#1a1a19",
    "#f0efec",
    "#c3c2b7",
    "#898781",
    "#2c2c2a",
    "#383835",
    "#ffffff",
    "#3987e5",
    "#898781",
    "#d95926",
    0.1,
    0.26,
)


def _n(value: float) -> str:
    text = f"{value:.1f}"
    return "0.0" if text == "-0.0" else text


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _signed(value: float, digits: int = 3) -> str:
    text = f"{value:+.{digits}f}"
    return text.replace("-", MINUS)


def _tick(value: float) -> str:
    text = f"{value:g}"
    return "0" if text in {"0", "-0"} else text.replace("-", MINUS)


class Svg:
    """An SVG document built element by element, every coordinate at one decimal."""

    def __init__(self, width: int, height: int, theme: Theme, title: str, desc: str) -> None:
        self.width, self.height, self.theme = width, height, theme
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc" '
            f'font-family="{FONT}">',
            f'<title id="title">{_escape(title)}</title>',
            f'<desc id="desc">{_escape(desc)}</desc>',
            f'<rect x="0.5" y="0.5" width="{width - 1}" height="{height - 1}" rx="10" '
            f'fill="{theme.surface}" stroke="{theme.ring}" stroke-opacity="0.1"/>',
        ]

    def add(self, part: str) -> None:
        self.parts.append(part)

    def text(
        self,
        x: float,
        y: float,
        text: str,
        *,
        fill: str,
        size: float = 12,
        weight: int = 400,
        anchor: str = "start",
        tabular: bool = False,
    ) -> None:
        extra = ' font-variant-numeric="tabular-nums"' if tabular else ""
        bold = f' font-weight="{weight}"' if weight != 400 else ""
        end = f' text-anchor="{anchor}"' if anchor != "start" else ""
        self.add(
            f'<text x="{_n(x)}" y="{_n(y)}" font-size="{size:g}" fill="{fill}"{bold}{end}{extra}>'
            f"{_escape(text.replace(APOSTROPHE, TYPOGRAPHIC))}</text>"
        )

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        stroke: str,
        width: float = 1,
        cap: str = "butt",
    ) -> None:
        round_cap = f' stroke-linecap="{cap}"' if cap != "butt" else ""
        self.add(
            f'<line x1="{_n(x1)}" y1="{_n(y1)}" x2="{_n(x2)}" y2="{_n(y2)}" stroke="{stroke}" '
            f'stroke-width="{width:g}"{round_cap}/>'
        )

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        *,
        fill: str,
        rx: float = 0,
        opacity: float = 1,
    ) -> None:
        alpha = f' fill-opacity="{opacity:g}"' if opacity != 1 else ""
        rounded = f' rx="{rx:g}"' if rx else ""
        self.add(
            f'<rect x="{_n(x)}" y="{_n(y)}" width="{_n(w)}" height="{_n(h)}"{rounded} '
            f'fill="{fill}"{alpha}/>'
        )

    def dot(
        self,
        x: float,
        y: float,
        *,
        fill: str,
        hollow: bool = False,
        ring: bool = True,
        r: float = 4.5,
    ) -> None:
        """A marker with its 2 px ring in the surface's colour, which `ring` false leaves off where
        the marker sits on its own interval; hollow, a ring of `fill` instead."""
        surface = self.theme.surface
        if hollow:
            self.add(
                f'<circle cx="{_n(x)}" cy="{_n(y)}" r="{r + 1:g}" fill="{surface}"/>'
                f'<circle cx="{_n(x)}" cy="{_n(y)}" r="{r - 1:g}" fill="{surface}" '
                f'stroke="{fill}" stroke-width="2"/>'
            )
        else:
            halo = f' stroke="{surface}" stroke-width="2"' if ring else ""
            self.add(f'<circle cx="{_n(x)}" cy="{_n(y)}" r="{r:g}" fill="{fill}"{halo}/>')

    @contextlib.contextmanager
    def group(self, title: str) -> Generator[None]:
        """Marks that hover as one, with `title` as their tooltip."""
        self.add(f"<g><title>{_escape(title)}</title>")
        yield
        self.add("</g>")

    def render(self) -> str:
        return "\n".join([*self.parts, "</svg>"]) + "\n"


@dataclass(frozen=True, slots=True)
class Scale:
    """A linear map from `domain` to the pixels `start` to `stop`."""

    lo: float
    hi: float
    start: float
    stop: float

    def __call__(self, value: float) -> float:
        return self.start + (value - self.lo) / (self.hi - self.lo) * (self.stop - self.start)


@dataclass(frozen=True, slots=True)
class Cell:
    """One family-rung of the record: the run that read it and its comparisons by rival."""

    run: str
    family: int
    k: int
    name: str
    found: dict[str, dict[str, Any]]
    claimed: bool

    @property
    def geo(self) -> bool:
        return next(iter(self.found.values()))["arm"] == "geo eb"

    @property
    def closest(self) -> dict[str, Any]:
        """The comparison whose 95 % interval reaches highest: the one nearest to the margin."""
        compared = [self.found[r] for r in record.SCORED if self.found[r]["mean"] is not None]
        return max(compared, key=lambda c: float(c["high"]))

    def label(self) -> str:
        rung = "k = 0" if self.k == 0 else f"tested rung (k = {self.k})"
        return f"Family {self.family} {self.name}, {rung}, {record.RUNS[self.run].name}"


def cells(records: record.Records) -> list[Cell]:
    """The record's family-rungs, as `readme_record.family_rungs` orders them: national families
    first, geo families after them, each family's k = 0 rung before its tested one."""
    claimed = {(r.family, r.k) for r in record._rows(records)}
    return [
        Cell(
            r.run,
            r.family,
            r.k,
            record.family_names(record.RUNS[r.run])[r.family],
            r.found,
            (r.family, r.k) in claimed,
        )
        for r in record.family_rungs(records)
    ]


def _interval(c: dict[str, Any]) -> str:
    return f"{_signed(float(c['mean']))} [{_signed(float(c['low']))}, {_signed(float(c['high']))}]"


def _hover(cell: Cell, c: dict[str, Any]) -> str:
    within = "within δ" if c["within"] else "not within δ = 0.05, no claim"
    return (
        f"{cell.label()}: mixdyn less {record.NAMES[c['rival']]} {_interval(c)} a euro, "
        f"{c['word']}, {within}, {c['complete']} worlds"
    )


def _axis(
    svg: Svg, x: Scale, top: float, bottom: float, ticks: list[float], *, labels: bool = True
) -> None:
    """Hairline gridlines at `ticks`, 0 on the axis's own rule, tick labels below `bottom`."""
    t = svg.theme
    for v in ticks:
        svg.line(x(v), top, x(v), bottom, stroke=t.axis if v == 0 else t.grid)
        if labels:
            svg.text(
                x(v), bottom + 16, _tick(v), fill=t.muted, size=11, anchor="middle", tabular=True
            )


def _beyond(svg: Svg, x: Scale, top: float, bottom: float, *, label: bool) -> None:
    """The region past δ washed in the losing pole's hue, δ's rule at its edge."""
    t = svg.theme
    svg.rect(x(DELTA), top, x.stop - x(DELTA), bottom - top, fill=t.loses, opacity=t.wash)
    svg.line(x(DELTA), top, x(DELTA), bottom, stroke=t.loses, width=1.5)
    if label:
        svg.text(x(DELTA) + 5, top + 12, "δ = 0.05", fill=t.ink2, size=11, weight=600)
        svg.text(x(DELTA) + 5, top + 26, "no claim", fill=t.ink2, size=11)


def _legend(svg: Svg, x: float, y: float) -> float:
    """The reader's three words and the wash past δ, left to right from `x`; returns its end."""
    t = svg.theme
    items = (
        ("beats", "mixdyn beats the rival"),
        ("ties", "ties"),
        ("loses to", "loses to it"),
    )
    for word, text in items:
        svg.dot(x + 5, y - 4, fill=t.word(word), hollow=word == "ties")
        svg.text(x + 15, y, text, fill=t.ink2, size=12)
        x += 15 + 6.0 * len(text) + 18
    svg.rect(x, y - 11, 16, 14, fill=t.loses, opacity=t.wash * 1.5)
    svg.line(x, y - 11, x, y + 3, stroke=t.loses, width=1.5)
    svg.text(x + 20, y, "past δ = 0.05: no claim", fill=t.ink2, size=12)
    return x + 20 + 6.0 * 23


def closest_rival(records: record.Records, theme: Theme) -> str:
    """Each family's two rungs side by side: mixdyn's arm less the rival nearest to the margin."""
    every = cells(records)
    families = sorted({(c.geo, c.family) for c in every})
    by = {(c.family, c.k != 0): c for c in every}
    national = [f for geo, f in families if not geo]
    geo = [f for g, f in families if g]

    pad, label_w, panel_w, rival_w, gap = 24, 228, 246, 124, 16
    width = pad + label_w + 2 * (panel_w + rival_w) + gap + pad
    row, head = 23, 172
    top = head + 22
    rows_y = {f: top + i * row for i, f in enumerate(national)}
    geo_top = top + len(national) * row + 30
    rows_y |= {f: geo_top + i * row for i, f in enumerate(geo)}
    bottom = geo_top + len(geo) * row
    height = int(bottom + 40)

    held = sum(c.claimed for c in every)
    svg = Svg(
        width,
        height,
        theme,
        "mixdyn against its closest rival on every family",
        f"For each of the {len(every)} family-rungs of M3, M4 and M5, mixdyn's plan less the plan "
        "of the rival whose 95 % interval reaches highest, in regret per euro of budget, lower "
        f"better. {held} lie within the non-inferiority margin of 0.05; family 4 at k = 0 does "
        "not.",
    )
    t = theme
    svg.text(
        pad,
        40,
        "mixdyn against its closest rival, family by family",
        fill=t.ink,
        size=17,
        weight=600,
    )
    sub = (
        "Regret per euro of budget, mixdyn's plan less the rival's: mean and paired 95 % interval "
        "over the simulated worlds both planned,",
        "for the rival whose interval reaches highest. Left of 0, mixdyn's plan lost less. The "
        "claim is non-inferiority: every interval left of δ.",
    )
    for i, line in enumerate(sub):
        svg.text(pad, 64 + 17 * i, line, fill=t.ink2, size=12.5)
    _legend(svg, pad, 118)

    x0 = pad + label_w
    panels = []
    for i, (tested, title) in enumerate(
        ((False, "History only (k = 0)"), (True, "With lift tests (the tested rung)"))
    ):
        left = x0 + i * (panel_w + rival_w + gap)
        x = Scale(-0.22, 0.2, left + 8, left + panel_w - 8)
        panels.append((tested, x, left))
        svg.text(left + 8, head - 6, title, fill=t.ink, size=13, weight=600)
        svg.text(left + panel_w + 10, head - 6, "closest rival", fill=t.muted, size=11)
        blocks = (
            (top - row / 2, top + (len(national) - 0.5) * row),
            (geo_top - row / 2, bottom - row / 2),
        )
        for y0, y1 in blocks:
            _beyond(svg, x, y0, y1, label=False)
            _axis(svg, x, y0, y1, [-0.2, -0.1, 0.0, 0.1, 0.2], labels=y1 == blocks[-1][1])
        svg.text(x(DELTA) + 4, top - row / 2 - 4, "δ", fill=t.ink2, size=12, weight=600)

    svg.text(pad, top - row / 2 - 4, "National families, M4 and M5", fill=t.muted, size=11)
    svg.text(pad, geo_top - row / 2 - 4, "Geo families, M3", fill=t.muted, size=11)
    for f, y in rows_y.items():
        name = by[f, False].name
        svg.text(pad + 18, y + 4, str(f), fill=t.muted, size=12, anchor="end", tabular=True)
        svg.text(pad + 26, y + 4, name, fill=t.ink, size=12)
        for tested, x, left in panels:
            cell = by[f, tested]
            c = cell.closest
            word = c["word"]
            mean, low, high = (float(c[k]) for k in ("mean", "low", "high"))
            with svg.group(_hover(cell, c)):
                svg.rect(left, y - row / 2, panel_w + rival_w, row, fill=t.surface, opacity=0)
                svg.line(x(low), y, x(high), y, stroke=t.word(word), width=2, cap="round")
                svg.dot(x(mean), y, fill=t.word(word), hollow=word == "ties", ring=False)
            rival = record.NAMES[c["rival"]]
            run = record.RUNS[cell.run].name
            svg.text(
                left + panel_w + 10,
                y + 4,
                rival,
                fill=t.ink2 if cell.claimed else t.ink,
                size=11.5,
                weight=400 if cell.claimed else 600,
            )
            if run != "M4" and not cell.geo:
                svg.text(
                    left + panel_w + rival_w - 2, y + 4, run, fill=t.muted, size=10.5, anchor="end"
                )
    for _, _, left in panels:
        svg.text(
            left + panel_w / 2,
            bottom - row / 2 + 34,
            "mixdyn less the rival, regret per euro",
            fill=t.muted,
            size=11,
            anchor="middle",
        )
    return svg.render()


def swarm(xs: list[float], d: float) -> list[float]:
    """A beeswarm: each point, from the left, at the offset nearest 0 that keeps it `d` from every
    point already placed."""
    placed: list[tuple[float, float]] = []
    ys = [0.0] * len(xs)
    for i in sorted(range(len(xs)), key=lambda i: (xs[i], i)):
        x = xs[i]
        near = [(px, py) for px, py in placed if abs(px - x) < d]
        candidates = {0.0}
        for px, py in near:
            h = math.sqrt(d * d - (px - x) ** 2)
            candidates |= {py + h, py - h}
        for y in sorted(candidates, key=lambda y: (abs(y), y)):
            if all((px - x) ** 2 + (py - y) ** 2 >= d * d - 1e-9 for px, py in near):
                ys[i] = y
                break
        placed.append((x, ys[i]))
    return ys


def every_comparison(records: record.Records, theme: Theme) -> str:
    """Every comparison of the record as one dot a family-rung, a row a rival."""
    every = cells(records)
    pad, label_w, plot_w, count_w = 24, 150, 590, 196
    width = pad + label_w + plot_w + 24 + count_w + pad
    left = pad + label_w
    x = Scale(-0.75, 0.2, left, left + plot_w)
    r, spacing = 4.5, 10.0

    rows = []
    for rival in record.SCORED:
        compared = [(cell, cell.found[rival]) for cell in every if cell.found[rival]["mean"]]
        ys = swarm([x(float(c["mean"])) for _, c in compared], spacing)
        reach = max(abs(y) for y in ys)
        rows.append((rival, compared, ys, max(2 * reach + 2 * r + 26, 92.0)))

    head = 148
    height = int(head + sum(h for *_, h in rows) + 56)
    total = sum(len(c) for _, c, _, _ in rows)
    svg = Svg(
        width,
        height,
        theme,
        "Every comparison of the record, rival by rival",
        f"{total} comparisons of mixdyn's plans with five rivals' over {len(every)} family-rungs: "
        "each dot one family-rung's mean difference in regret per euro, coloured by Hall's "
        "reader's word.",
    )
    t = theme
    svg.text(pad, 40, "Every comparison, rival by rival", fill=t.ink, size=17, weight=600)
    sub = (
        f"Each dot is one of the {len(every)} family-rungs: mixdyn's mean regret per euro less the "
        "rival's, over the simulated worlds both planned.",
        "Its colour is the word of Hall's reader at one-sided 0.025. Left of 0, mixdyn's plan lost "
        "less; past δ = 0.05, the claim fails.",
    )
    for i, line in enumerate(sub):
        svg.text(pad, 64 + 17 * i, line, fill=t.ink2, size=12.5)
    _legend(svg, pad, 118)

    y0 = head
    plot_top = y0
    plot_bottom = y0 + sum(h for *_, h in rows)
    _beyond(svg, x, plot_top, plot_bottom, label=True)
    _axis(svg, x, plot_top, plot_bottom, [-0.6, -0.4, -0.2, 0.0, 0.2])
    for rival, compared, ys, h in rows:
        mid = y0 + h / 2
        if y0 > head:
            svg.line(pad, y0, width - pad, y0, stroke=t.grid)
        svg.text(pad, mid - 2, record.NAMES[rival], fill=t.ink, size=13, weight=600)
        failed = len(every) - len(compared)
        svg.text(pad, mid + 15, f"{len(compared)} family-rungs", fill=t.muted, size=11)
        for (cell, c), dy in zip(compared, ys, strict=True):
            word = c["word"]
            with svg.group(_hover(cell, c)):
                svg.dot(x(float(c["mean"])), mid + dy, fill=t.word(word), hollow=word == "ties")
            if not c["within"]:
                svg.text(
                    x(float(c["mean"])),
                    mid + dy - 10,
                    f"{cell.name}, k = {cell.k}",
                    fill=t.ink2,
                    size=10.5,
                    anchor="middle",
                )
        said = [c["word"] for _, c in compared]
        lines = [(w, said.count(w)) for w in WORDS if said.count(w)]
        beyond = sum(not c["within"] for _, c in compared)
        cx = left + plot_w + 24
        ly = mid - 7.5 * (len(lines) + bool(failed)) + 11
        for word, count in lines:
            svg.dot(cx + 5, ly - 4, fill=t.word(word), hollow=word == "ties", r=4)
            text = word + (f" ({beyond} past δ)" if word == "loses to" and beyond else "")
            svg.text(cx + 16, ly, text, fill=t.ink2, size=11.5)
            svg.text(
                cx + count_w, ly, str(count), fill=t.ink, size=11.5, anchor="end", tabular=True
            )
            ly += 15
        if failed:
            svg.text(cx + 16, ly, "not compared", fill=t.muted, size=11.5)
            svg.text(
                cx + count_w, ly, str(failed), fill=t.muted, size=11.5, anchor="end", tabular=True
            )
        y0 += h
    svg.text(
        x(-0.275),
        plot_bottom + 36,
        "mixdyn less the rival, mean regret per euro",
        fill=t.muted,
        size=11,
        anchor="middle",
    )
    return svg.render()


COST_RUNS = ("m4", "m5")
COST_ARMS = (
    ("margin", "mixdyn", "history only"),
    ("anchored", "mixdyn", "with lift tests"),
    ("orbit", "Orbit", ""),
    ("pymc-marketing", "PyMC-Marketing", ""),
    ("meridian", "Meridian", ""),
)


@dataclass(frozen=True, slots=True)
class Cost:
    """One arm's processes over M4 and M5: their count, their mean CPU time and the largest peak
    memory of any of them."""

    name: str
    rung: str
    rival: bool
    processes: int
    cpu_seconds: float
    peak_gib: float


def costs(records: record.Records) -> list[Cost]:
    """The national claim's arms and the rivals over M4 and M5, which ran on one machine class: the
    mean is the runs' CPU time over their processes, exact from the records' sums."""
    out = []
    for arm, name, rung in COST_ARMS:
        rows = [records[run]["compute"]["arms"][arm] for run in COST_RUNS]
        processes = sum(r["processes"] for r in rows)
        out.append(
            Cost(
                name,
                rung,
                rows[0]["rival"],
                processes,
                sum(r["core_hours"] for r in rows) * 3600.0 / processes,
                max(r["peak_gib"]["most"] for r in rows),
            )
        )
    return out


def _count(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def _bar(svg: Svg, x0: float, x1: float, y: float, h: float, fill: str) -> None:
    """A bar from the baseline `x0` to `x1`, square at the baseline, its data end rounded 4 px."""
    r = min(4.0, (x1 - x0) / 2)
    top, bottom = y - h / 2, y + h / 2
    svg.add(
        f'<path d="M{_n(x0)} {_n(top)}H{_n(x1 - r)}Q{_n(x1)} {_n(top)} {_n(x1)} {_n(top + r)}'
        f'V{_n(bottom - r)}Q{_n(x1)} {_n(bottom)} {_n(x1 - r)} {_n(bottom)}H{_n(x0)}Z" '
        f'fill="{fill}"/>'
    )


def cost(records: record.Records, theme: Theme) -> str:
    """Each arm's mean CPU time a process and largest peak memory, mixdyn's arms against the
    rivals', over M4 and M5."""
    rows = costs(records)
    models = sorted(
        {
            m
            for run in COST_RUNS
            for f in records[run]["compute"]["families"].values()
            for m in f["cpu_model"]
        }
    )
    base = max(c.cpu_seconds for c in rows if not c.rival)
    pad, label_w, cpu_w, gap, mem_w = 24, 200, 420, 36, 296
    width = pad + label_w + cpu_w + gap + mem_w + pad
    row, bar, top = 40, 18, 150
    ys, y = [], top + row / 2
    for i, c in enumerate(rows):
        if i and c.rival != rows[i - 1].rival:
            y += 14
        ys.append(y)
        y += row
    bottom = ys[-1] + row / 2
    height = int(bottom + 34)
    t = theme
    rungs = record.family_rungs(records)
    held = len(record._rows(records))
    title = (
        f"Plans within δ of theirs on {held} of {len(rungs)} family-rungs, at a fraction of the "
        "compute"
    )
    svg = Svg(
        width,
        height,
        t,
        title,
        "The mean CPU time of a process and the largest peak memory of any, for mixdyn's two arms "
        "and for Orbit, PyMC-Marketing and Meridian, over the runs M4 and M5 on one machine class.",
    )
    svg.text(pad, 40, title, fill=t.ink, size=17, weight=600)
    sub = (
        f"M4 and M5 on one machine class ({', '.join(models)}), one thread a process, each "
        "rival at its recommended settings.",
        "A rival's process fits and plans one world at one rung; a process of mixdyn's "
        "history-only arm plans a world at up to two rungs.",
        "Each figure is the process's own scope's, spent on a shared box: what each tool cost "
        "there, not a speed benchmark.",
    )
    for i, line in enumerate(sub):
        svg.text(pad, 64 + 17 * i, line, fill=t.ink2, size=12.5)

    cx = Scale(0.0, 340.0, pad + label_w, pad + label_w + cpu_w)
    mx = Scale(0.0, 18.0, pad + label_w + cpu_w + gap, width - pad)
    svg.text(cx.start, top - 14, "CPU time a process, mean", fill=t.ink, size=13, weight=600)
    svg.text(
        mx.start, top - 14, "Peak memory, the largest process", fill=t.ink, size=13, weight=600
    )
    for scale, ticks, unit in ((cx, (0, 100, 200, 300), " s"), (mx, (0, 5, 10, 15), " GiB")):
        for v in ticks:
            svg.line(scale(v), top, scale(v), bottom, stroke=t.axis if v == 0 else t.grid)
            svg.text(
                scale(v),
                bottom + 16,
                f"{v}{unit if v else ''}",
                fill=t.muted,
                size=11,
                anchor="middle",
                tabular=True,
            )
    for c, y in zip(rows, ys, strict=True):
        fill = t.ties if c.rival else t.beats
        svg.text(pad, y - 1, c.name, fill=t.ink, size=13, weight=600)
        detail = f"{c.rung}, " if c.rung else ""
        svg.text(pad, y + 14, f"{detail}{_count(c.processes)} processes", fill=t.muted, size=11)
        seconds = f"{c.cpu_seconds:.1f} s" if c.cpu_seconds < 100 else f"{c.cpu_seconds:.0f} s"
        with svg.group(
            f"{c.name}{', ' + c.rung if c.rung else ''}: {_count(c.processes)} processes over M4 "
            f"and M5, {c.cpu_seconds:.1f} CPU seconds a process on average, peak memory "
            f"{c.peak_gib:.2f} GiB at most"
        ):
            _bar(svg, cx(0), cx(c.cpu_seconds), y, bar, fill)
            _bar(svg, mx(0), mx(c.peak_gib), y, bar, fill)
        svg.text(cx(c.cpu_seconds) + 6, y + 4, seconds, fill=t.ink, size=12, tabular=True)
        if c.rival:
            svg.text(
                cx(c.cpu_seconds) + 12 + 7.0 * len(seconds),
                y + 4,
                f"{c.cpu_seconds / base:.0f}{TIMES} mixdyn's",
                fill=t.ink2,
                size=12,
            )
        svg.text(
            mx(c.peak_gib) + 6, y + 4, f"{c.peak_gib:.2f} GiB", fill=t.ink, size=12, tabular=True
        )
    return svg.render()


def example() -> dict[str, Any]:
    """The README's first example run as the README holds it, and its names after it."""
    readme = record.README.read_text(encoding="utf-8")
    code = re.findall(r"^```python\n(.*?)^```$", readme, flags=re.DOTALL | re.MULTILINE)[0]
    namespace: dict[str, Any] = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(code, "README.md", "exec"), namespace)
    return namespace


def first_plan(held: dict[str, Any], theme: Theme) -> str:
    """The example's margins against its truth, and its plan against the status quo."""
    fit, truth, quo, plan = held["fit"], held["truth"], held["status_quo"], held["plan"].spend
    low, high = (
        np.array([getattr(r, end) for r in held["limits"].values()]) for end in ("least", "most")
    )
    sd = np.sqrt(np.diag(fit.cov))
    pad, label_w, panel_w, gap, value_w = 24, 86, 360, 48, 112
    width = pad + label_w + panel_w + gap + panel_w + value_w + pad
    row, top = 40, 158
    height = top + 3 * row + 34
    t = theme
    svg = Svg(
        width,
        height,
        t,
        "The README's first plan, drawn",
        "Three channels' margins as fitted on two simulated years of weekly history, with their 95 "
        "% intervals, against the margins the history was drawn with; and next quarter's weekly "
        "plan against the status quo, inside half to twice each channel's status quo.",
    )
    svg.text(pad, 40, "The first plan, drawn", fill=t.ink, size=17, weight=600)
    sub = (
        "The example above, on two simulated years of weekly history: each channel's margin as "
        "fitted, against the margin the history was drawn with,",
        "and next quarter's weekly spend at the same budget, filled along each channel's curve "
        "from its shrunk margin, inside half to twice the status quo.",
    )
    for i, line in enumerate(sub):
        svg.text(pad, 64 + 17 * i, line, fill=t.ink2, size=12.5)

    lx = pad + label_w
    rx = lx + panel_w + gap
    m = Scale(-1.25, 4.25, lx + 8, lx + panel_w - 8)
    s = Scale(0.0, 300.0, rx + 8, rx + panel_w - 8)
    svg.text(
        lx + 8,
        top - 30,
        "Margin: sales a unit of weekly spend returns",
        fill=t.ink,
        size=13,
        weight=600,
    )
    svg.text(rx + 8, top - 30, "Weekly spend, next quarter", fill=t.ink, size=13, weight=600)
    key = top - 14
    svg.line(lx + 8, key - 4, lx + 26, key - 4, stroke=t.beats, width=2, cap="round")
    svg.dot(lx + 17, key - 4, fill=t.beats, ring=False)
    svg.text(lx + 32, key, "fitted, 95 % interval", fill=t.ink2, size=11.5)
    svg.line(lx + 166, key - 11, lx + 166, key + 3, stroke=t.ink, width=2)
    svg.text(lx + 174, key, "truth", fill=t.ink2, size=11.5)
    svg.rect(rx + 8, key - 9, 22, 10, fill=t.beats, rx=5, opacity=t.band)
    svg.text(rx + 36, key, "allowed", fill=t.ink2, size=11.5)
    svg.dot(rx + 101, key - 4, fill=t.ties)
    svg.text(rx + 110, key, "status quo", fill=t.ink2, size=11.5)
    svg.dot(rx + 189, key - 4, fill=t.beats)
    svg.text(rx + 198, key, "plan", fill=t.ink2, size=11.5)

    y0, y1 = top, top + 3 * row
    for v in range(-1, 5):
        svg.line(m(v), y0, m(v), y1, stroke=t.axis if v == 0 else t.grid)
        svg.text(m(v), y1 + 16, _tick(v), fill=t.muted, size=11, anchor="middle", tabular=True)
    for v in (0, 100, 200, 300):
        svg.line(s(v), y0, s(v), y1, stroke=t.axis if v == 0 else t.grid)
        svg.text(s(v), y1 + 16, _tick(v), fill=t.muted, size=11, anchor="middle", tabular=True)
    for i in range(3):
        y = top + (i + 0.5) * row
        svg.text(pad, y + 4, f"Channel {i + 1}", fill=t.ink, size=12)
        mean, lo, hi = fit.mean[i], fit.mean[i] - 1.96 * sd[i], fit.mean[i] + 1.96 * sd[i]
        with svg.group(
            f"Channel {i + 1}: fitted margin {mean:.2f}, 95 % interval [{lo:.2f}, {hi:.2f}], "
            f"truth {truth[i]:.2f}"
        ):
            svg.line(m(lo), y, m(hi), y, stroke=t.beats, width=2, cap="round")
            svg.dot(m(mean), y, fill=t.beats, ring=False)
            svg.line(m(truth[i]), y - 9, m(truth[i]), y + 9, stroke=t.ink, width=2)
        with svg.group(
            f"Channel {i + 1}: status quo {quo[i]:.1f} a week, plan {plan[i]:.1f}, allowed "
            f"{low[i]:.1f} to {high[i]:.1f}"
        ):
            svg.rect(
                s(low[i]), y - 6, s(high[i]) - s(low[i]), 12, fill=t.beats, rx=6, opacity=t.band
            )
            svg.line(s(quo[i]), y, s(plan[i]), y, stroke=t.beats, width=2, cap="round")
            svg.dot(s(quo[i]), y, fill=t.ties)
            svg.dot(s(plan[i]), y, fill=t.beats)
        svg.text(
            rx + panel_w + 6,
            y + 4,
            f"{quo[i]:.0f} → {plan[i]:.0f}",
            fill=t.ink2,
            size=12,
            tabular=True,
        )
    return svg.render()


CHARTS: dict[str, Callable[[record.Records, dict[str, Any], Theme], str]] = {
    "closest-rival": lambda records, _, theme: closest_rival(records, theme),
    "every-comparison": lambda records, _, theme: every_comparison(records, theme),
    "first-plan": lambda _, held, theme: first_plan(held, theme),
    "cost": lambda records, _, theme: cost(records, theme),
}


def render(records: record.Records, held: dict[str, Any]) -> dict[str, str]:
    """Each chart's file name and text, both schemes."""
    return {
        f"{name}-{theme.name}.svg": draw(records, held, theme)
        for name, draw in CHARTS.items()
        for theme in (LIGHT, DARK)
    }


def main() -> int:
    ASSETS.mkdir(exist_ok=True)
    for name, text in render(record.records(), example()).items():
        (ASSETS / name).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
