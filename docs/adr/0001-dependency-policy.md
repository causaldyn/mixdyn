# 0001. Runtime dependencies: general packages only, checked against the lock

Status: accepted, 2026-10-01; its last decision, installation from the git source alone, is
superseded on 2026-11-20 by ADR 0095: the engine is published on PyPI

## Context

The author's terms for the engine: "as much our own as possible, as exact and as fast as possible".
Fewer dependencies here means fewer *special-purpose* packages. General ones are fine.

The packages it competes with are special-purpose MMM stacks: Meridian, PyMC-Marketing, Robyn. So
are the inference and statistics frameworks under them: Stan, PyMC, NumPyro, BlackJAX,
statsmodels, ArviZ.

Depending on any of them would make the engine's numbers theirs. It would also tie its releases to
their APIs, and its accuracy to choices it cannot inspect.

A package does not need to be named in `pyproject.toml` to arrive: a general package can bring a
special one in transitively. A reviewer reading the `pyproject.toml` diff would not see it.

## Decision

- **Runtime packages are an allow-list** in `allowed-runtime-packages.txt`:
  - NumPy and SciPy;
  - causal-hybrid-control with its runtime closure, which is JAX and the solvers built on it;
  - scikit-learn with its closure, which the author counts as general.
- **CI checks the closure that `uv.lock` resolves**, not the direct dependencies.
  - `scripts/check_runtime_deps.py` walks from the project through every dependency, its own
    extras, and the extras each edge requests.
  - It ignores development groups. It follows edges whatever their environment marker says.
  - It fails naming each package outside the list and the path that brought it in.
- **Special-purpose packages are oracles only.** They run in a scratch environment
  (`uv run --with …`) or in the benchmark repository, never in this lock.
- **Every addition is the author's decision**: to the allow-list, to the lock, to a bench arm's
  environment, or to a scratch library. Tools already in use (pytest, hypothesis, ruff, ty) are
  exempt.
- **The engine installs from its git source, never from a package index.**
  - An extra index would let a public package of the same name win the resolution.
  - The `Private :: Do Not Upload` classifier makes PyPI refuse an accidental upload.

## Consequences

- Samplers, conjugate blocks and diagnostics are written here, and each needs an independent
  oracle before anyone uses it: a closed form, another implementation, coverage over seeds, or
  simulation-based calibration. That is more code, and in exchange every numerical choice is
  visible and testable.
- The gate errs strict:
  - A marker-gated package that only some platform would install still has to be on the list.
  - Allowing a package allows its closure, and each new transitive package still shows up as a
    named failure until it is added.

## Alternatives considered

- **Review only, no gate.** Rejected: transitive arrivals do not show in a `pyproject.toml` diff.
- **A deny-list of known special packages.** Rejected: the set is open, so a new package would pass
  in silence. The allow-list fails closed.
- **Vendoring the needed parts of a special package.** Rejected: it keeps their numerics and drops
  their maintenance.
