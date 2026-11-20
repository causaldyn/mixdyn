"""The tools' arms: the bench's own scripts, each run from the bench's tree in the environment the
bench pins for it.

A Python tool runs through `uv run --no-project --python 3.12 --with-requirements
scripts/<arm>.txt python scripts/<arm>.py`, which builds the environment the bench's requirements
pin; Robyn through `Rscript --vanilla scripts/robyn_arm.R` over the genre ranges the bench
pre-registered, on the R library and nevergrad's Python environment a run declares. Each script
fits every world of `--worlds` that `--out` does not hold and writes one JSON record a world;
`scorecard.run` hands it one world's directory as both and moves the record into the run.

**Tuned arms.** Beside each rival's recommended settings, one tuned configuration a rival, each a
recorded arm of its own so that the recommended arm's records stay what they were: `robyn tuned`,
the genre ranges with Robyn's time-series validation on, fixed before any scored world; and
`meridian tuned` and `pymc-marketing tuned`, each one candidate of its rival's grid (`GRIDS`),
the one `scorecard.tuning` chose on the development worlds by the rival's own criterion and wrote
to `tuned.json` beside this module. A tuned arm whose rival the tuning has not chosen for is
refused where a run declares it (`Tool.environment`), and so is one whose choice adds no option to
its rival's arm, the recommended settings: its column would fit the recommended column again.
"""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from scorecard.layout import Declaration

ORBIT_INDEX = (
    "--extra-index-url",
    "https://download.pytorch.org/whl/cpu",
    "--index-strategy",
    "unsafe-best-match",
)
"""Where Orbit's environment finds PyTorch's CPU wheels, as the bench's requirements ask."""


@dataclass(frozen=True, slots=True)
class Tool:
    """A tool's script: `command`, run from the bench's tree, to which one world's directory is
    given; `r` where it runs in R, on the R library and nevergrad's environment a run declares;
    `pins`, the bench's files its records depend on beyond the modules its Python files import
    from the bench's `scripts/` (`fingerprint`): its script and the lock of its environment."""

    command: tuple[str, ...]
    r: bool = False
    pins: tuple[str, ...] = ()
    unset: str | None = None
    """Why the tool cannot run yet: a tuned arm whose configuration the tuning has not chosen."""

    def environment(self, declaration: Declaration) -> dict[str, str]:
        """What the script's process reads beyond the harness's own environment.

        Raises:
            ValueError: on a tuned arm the tuning has not chosen, or a tool in R where the run
                declares no R library.
        """
        return self.paths(declaration.r_library, declaration.nevergrad)

    def paths(self, r_library: str | None, nevergrad: str | None) -> dict[str, str]:
        """What the script's process reads beyond the harness's own environment, on the R library
        and nevergrad's environment given, which a tool in R needs.

        Raises:
            ValueError: on a tuned arm the tuning has not chosen, or a tool in R without them.
        """
        if self.unset is not None:
            raise ValueError(self.unset)
        if not self.r:
            return {}
        if r_library is None or nevergrad is None:
            msg = "Robyn runs on the R library and nevergrad's environment the run declares"
            raise ValueError(msg)
        return {
            "R_LIBS": r_library,
            "R_LIBS_USER": r_library,
            "RETICULATE_PYTHON": str(Path(nevergrad) / "bin" / "python"),
        }


def pinned(script: str, *options: str, index: tuple[str, ...] = ()) -> Tool:
    """The bench's `scripts/<script>.py` in the environment `scripts/<script>.txt` pins."""
    requirements = ("--with-requirements", f"scripts/{script}.txt")
    run = ("uv", "run", "--no-project", "--python", "3.12", *index, *requirements)
    pins = (f"scripts/{script}.py", f"scripts/{script}.txt")
    return Tool((*run, "python", f"scripts/{script}.py", *options), pins=pins)


def fingerprint(tool: Tool, show: Callable[[str], bytes | None]) -> dict[str, str]:
    """The SHA-256 of each bench file `tool`'s records depend on, by its path: its `pins` and every
    module of the bench's `scripts/` its Python files import, and theirs in turn; `show` reads a
    file of the bench at one commit, None where it holds none, which is recorded as such."""
    out: dict[str, str] = {}
    todo = list(tool.pins)
    while todo:
        path = todo.pop()
        if path in out:
            continue
        held = show(path)
        out[path] = "absent" if held is None else hashlib.sha256(held).hexdigest()
        if held is None or not path.endswith(".py"):
            continue
        for node in ast.walk(ast.parse(held)):
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                names = [node.module.split(".")[0]]
            else:
                continue
            todo += [f"scripts/{n}.py" for n in names if show(f"scripts/{n}.py") is not None]
    return out


RESERVED: dict[str, float] = {
    "robyn": 3.0,
    "pymc-marketing": 1.5,
    "pymc-marketing shared": 1.5,
    "pymc-marketing channel": 1.5,
    "meridian": 1.2,
    "robyn tuned": 3.0,
    "pymc-marketing tuned": 1.5,
    "meridian tuned": 1.2,
}
"""The CPUs a stage reserves for one process of a tool that takes more than its one thread, from its
CPU-seconds over its wall in M2's run, a process's median: Robyn 2.88, on the cores its log names;
PyMC-Marketing 1.52 (shared 1.36, by channel 1.56), over its sampler's chains; Meridian 1.18. Every
other arm computes on one thread. A tuned arm reserves its rival's."""


GRIDS: dict[str, dict[str, tuple[str, ...]]] = {
    "meridian": {
        f"{prior} {knots}": (
            *(("--media-prior-type", prior) if prior != "roi" else ()),
            *(("--knot-weeks", knots.removeprefix("knots-")) if knots != "aks" else ()),
        )
        for prior in ("roi", "mroi", "contribution")
        for knots in ("aks", "knots-8", "knots-13")
    },
    "pymc-marketing": {
        f"{saturation} {intercept}": (
            *(("--saturation", saturation) if saturation != "logistic" else ()),
            *(("--intercept", intercept) if intercept != "static" else ()),
        )
        for saturation in ("logistic", "hill")
        for intercept in ("static", "varying")
        if (saturation, intercept) != ("hill", "varying")  # past COST_CAP
    },
}
"""Each tuned rival's candidates, by name: the options each adds to the rival's arm. The first of
each is the rival's recommended settings: Meridian's media prior type and the most weeks between
two knots (automatic knot selection, `aks`, as recommended), PyMC-Marketing's saturation and
intercept. Every candidate fits within `COST_CAP`."""
COST_CAP = 2.0
"""The most CPU a fit a tuned configuration may cost, as a multiple of its rival's recommended
settings' fit, alike for every rival: a limitation of the tuning, which searches only what a
scored run can afford. Measured at full samplers on one world of 156 weeks, each against the
recommended settings' fit on the same machine: PyMC-Marketing's Hill saturation with a varying
intercept, 7.7 times, left the grid; its Hill saturation with a static intercept fitted at 1.7
times, the logistic with a varying intercept at 1.6. Meridian's `mroi` and `contribution` priors
with automatic knot selection, and `mroi` with knots 8 weeks apart, fitted at 0.65 to 0.71 times
its recommended settings, every fit holding out its last tenth of weeks; its other candidates of
fixed knots hold 13 or 21 knots where the selection held 36 on that world and 36 to 55 on M2's
national worlds, so none of its nine is past the cap."""
RECOMMENDED = {"meridian": "roi aks", "pymc-marketing": "logistic static"}
TUNED = Path(__file__).with_name("tuned.json")
"""The tuning's choice of each rival's candidate (`scorecard.tuning select --write`), committed
before any scored world."""


def chosen(rival: str) -> str | None:
    """The candidate of `rival`'s grid the tuning chose (`TUNED`), None where it has chosen none."""
    held = json.loads(TUNED.read_text()).get(rival) if TUNED.exists() else None
    return None if held is None else held["candidate"]


def recommended(rival: str) -> bool:
    """Whether the tuning chose a candidate that adds no option to `rival`'s arm, its recommended
    settings: a run declares no tuned column of it, the recommended column being the tuned one."""
    candidate = chosen(rival)
    return candidate is not None and not GRIDS[rival][candidate]


def tuned(rival: str, base: Tool) -> Tool:
    """`base` with the options of the candidate the tuning chose for `rival`, or a tool that is
    refused where a run declares it: where the tuning has chosen none, or chose the recommended
    settings, whose column the run fits already (`recommended`)."""
    candidate = chosen(rival)
    if candidate is None:
        return Tool((), unset=f"the tuning has chosen no configuration of {rival}: see {TUNED}")
    if recommended(rival):
        return Tool(
            (),
            unset=(
                f"the tuning chose {rival}'s recommended settings, {candidate}: its recommended "
                f"column is its tuned one, so a run declares no `{rival} tuned`"
            ),
        )
    return Tool((*base.command, *GRIDS[rival][candidate]), base.r, base.pins)


TOOLS: dict[str, Tool] = {
    "pymc-marketing": pinned("pymc_marketing_arm", "--setting", "static"),
    "pymc-marketing shared": pinned("pymc_marketing_arm", "--setting", "shared"),
    "pymc-marketing channel": pinned("pymc_marketing_arm", "--setting", "channel"),
    "meridian": pinned("meridian_arm"),
    "orbit": pinned("orbit_arm", index=ORBIT_INDEX),
    "robyn": Tool(
        ("Rscript", "--vanilla", "scripts/robyn_arm.R", "--ranges", "genre"),
        r=True,
        pins=("scripts/robyn_arm.R", "scripts/robyn_arm.lock.txt", "scripts/geo_export.py"),
    ),
}
TOOLS |= {
    "meridian tuned": tuned("meridian", TOOLS["meridian"]),
    "pymc-marketing tuned": tuned("pymc-marketing", TOOLS["pymc-marketing"]),
    "robyn tuned": Tool(
        (*TOOLS["robyn"].command, "--ts-validation"), r=True, pins=TOOLS["robyn"].pins
    ),
}
"""The tools' arms, by the name a declaration gives them."""
