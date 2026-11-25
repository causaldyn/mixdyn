# 0103. What a release may change of the surface the last one shipped

Status: accepted, 2026-11-25; amends ADR 0062's rule for versions and ADR 0102's snapshot test:
each line of the snapshot at the last release tag stays, or changes as its kind of change allows;
the check that public names take only public classes follows a `type` alias to its value. Adds
twelve public classes and documents the fifteen entries left pending; moves no default and no saved
file's class.

## Context

ADR 0102 holds the public surface to `tests/public_api.txt`, one line per function's signature,
class, field, member and constant. Its writer refuses to drop a name before the release its
package's `_DEPRECATED` names. Three gaps were left for 1.0, which freezes the surface:

- **The snapshot is compared with itself.** The writer reads the file it is about to overwrite,
  not what a release shipped. A changed signature is written into the snapshot and passes, and so
  does a name dropped once the file is written from nothing. ADR 0062's removal rule was held for
  names that a package's `_DEPRECATED` lists, and by review for the rest.
- **A changed line is not always a break.** Read line by line, any edit of a line removes it. But
  a function that gains a keyword parameter with a default changes its line and breaks no caller.
  ADR 0102 lets a planner's default move in a minor release, with an ADR and a *Changed* entry.
  `_DEPRECATED` holds names, not parameters, so a changed signature cannot be listed there at all.
- **The check that public names take only public classes read an annotation's names, not what an
  alias stands for.** `BayesSpec.likelihood` is a `Likelihood`, an alias of `Normal | StudentT |
  LogNormal`, and none of the three was public. The same held for `BayesSpec`'s and
  `BayesGeoSpec`'s curve, `BayesMedia`'s prior, the Bayesian fits' start and coordinates, and the
  frame that `read_panel`'s `ColumnData` names. A caller could build these arguments only from a
  module path.

Besides, `tests/test_public_names.py` listed fifteen entries as not documented yet.

## Decision

### The last release's snapshot holds the surface

`tests/test_public_api.py` reads `tests/public_api.txt` at the newest release tag (`v*`) that the
checkout's commit contains, a tag on that commit itself excepted, and compares each of its lines
with the surface now. A line that is not there unchanged is of one of four kinds:

| Kind | What changed | Allowed |
|---|---|---|
| Left | The name is gone. | Where that release held the name leaving, from the release it named; from 1.0, in a major release only. |
| Extended | A function or a method gains parameters with defaults, keyword-only or after its positional ones; a parameter or a field gains a default. | In any release. |
| Revalued | Only a value moves: a parameter's or a field's default, a constant's or a class attribute's value. | From the next minor release: the version's major or minor is past the release's. |
| Changed | Anything else: a parameter removed, renamed, reordered, retyped or made required; a return type; a field's type; a member gone; a decorator, a base, an alias's value. | In a major release from 1.0 only: the release compared with is 1.0 or later, and the version's major is past its. |

- **Before 1.0, a public name changes only by extension or by a moved value.** A name whose
  signature must change takes a new name, and the old name leaves by the removal rule. So 1.0.0
  holds what 0.3.0 shipped, less the two names that leave in 1.0.0.
- **The version is the package metadata's**, as the removal rule's test reads it. A change that
  needs a minor release passes once `pyproject.toml` names that release.
- **Git and the release tags are read where the test runs.** Where no release tag with a snapshot
  is reachable, as in a shallow clone, the test skips and says so.

### The check of the classes public names take follows aliases

Each name in an annotation that is a `type` alias is followed to the names of its value, in the
module that defines the alias, and a public alias's own value is read too. The twelve classes this
found join an `__all__`, each in the module that defines it, so no `REGISTRY` key moves:
`RoiPrior`, `HalfNormalRoi`, `Normal`, `StudentT`, `LogNormal`, `HillCurve`, `TanhCurve`,
`UniformStart`, `PathfinderStart`, `NaturalCoordinates` and `QRCoordinates` in `mixdyn.models`, and
`ColumnFrame` in `mixdyn.data`. The aliases stay at their module paths, as `ReturnPrior` did when
ADR 0102 made `BayesMedia` public.

### Every public name is documented

The fifteen entries carry a docstring, or the string after their alias, and the list of names not
documented yet is gone: a public name without documentation fails `tests/test_public_names.py`.

## Consequences

- A release cannot drop or retype a name its predecessor shipped without the test naming the line
  and its kind of change. Dropping the two names that leave in 1.0.0 passes once the version is
  1.0.0.
- A moved default, a planner's included, fails until the change names its minor release in
  `pyproject.toml`. The rest of ADR 0102's rule for a planner's default, an ADR and a *Changed*
  entry, is still held by review.
- CI checks out one commit and no tags, so the test skips there until the test jobs' checkout
  fetches the tags (`fetch-depth: 0`). Until then, it runs only in a full clone.
- The check reads text. A parameter type that widens, `float` to `float | None`, breaks no caller
  but reads as changed. A dataclass field added without a default breaks a caller who builds the
  class, but adds a line and drops none, so it passes: the snapshot holds the fields, not the
  `__init__` that a dataclass writes from them.

## Alternatives considered

- **Every changed line read as a removal**, which must name a replacement in `_DEPRECATED`.
  Rejected: from 1.0 a keyword parameter with a default could come only in a major release, and a
  planner's default could not move in the minor release ADR 0102 allows.
- **A copy of the released snapshot in the repository**, so that a shallow clone runs the check.
  Rejected: it repeats what the tag holds, and a release commit that does not refresh it holds the
  next release to the one before it.
- **griffe's breaking-change check.** Rejected: it is a new dependency, and it counts a moved
  default as a break, which ADR 0102 allows in a minor release.
