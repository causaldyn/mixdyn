# 0062. The file format and the public surface

Status: accepted, 2026-11-01; amended 2026-11-06: `docs/api/index.md` lists the public surface,
written by `scripts/api_index.py` and held to it by a test, and documents no name by listing it;
amended 2026-11-25 by ADR 0102: every module's `__all__` is public, and `tests/test_public_api.py`
holds the surface to a snapshot of its signatures and to the removal rule; amended 2026-11-25 by
ADR 0103: what a release may change of the snapshot the last release shipped, by kind of change,
and every public name is documented

## Context

The engine is about to be read by people who did not write it (ADR 0054). Two things break first
for the users of a modelling package once it is public: a model saved by one version will not load
in the next, and a name a script imports is renamed or gone. Neither had a rule here. A fit lived
only in the process that drew it, unless its caller pickled it, and a pickle loads only while every
class it names keeps its module, its name and its fields, runs whatever code the file names, and
says nothing when it loads into a class that changed. The names a caller may rely on were never
listed: each package has an `__all__`, but no test read them, and the engines' own entry points
sit at module paths that no `__all__` names.

The data the engines read had the same gap at its edge. `read_panel` checked every value, but not
that the dates were spaced as the cadence said: daily dates read at a weekly cadence failed as six
off-grid errors a week, none naming the remedy. A panel of a few weeks for many channels read
without a word.

## Decision

### The file format

- **`mixdyn.api.serialize.save(obj, path)` writes, and `load(path)` reads, one format**: a zip
  archive holding `manifest.json` and one `.npy` file per array under `arrays/`.
- **The manifest** names the format (`"format": "mixdyn"`), its version (`"format_version"`, an
  integer), the versions of mixdyn and causal-hybrid-control that wrote the file, and the object
  as a tree of JSON nodes. JSON's own values carry `None`, booleans, integers, strings and finite
  floats, which JSON round-trips exactly. Every other value is an object with one `$`-key: a
  non-finite float, a tuple, a list, a frozen set, a mapping (a `dict` or a `MappingProxyType`, the
  kind it was), a NumPy scalar, an array (an index into `arrays/`, with whether it was a JAX array
  and whether it was read-only), a date, an enum member by its name, or a dataclass by its
  registered name and the fields its `__init__` takes.
- **Nothing is pickled.** Arrays are written and read with `allow_pickle=False`, and an array of
  Python objects is refused at `save`. Loading a file runs no code the file names: it builds only
  the classes of `REGISTRY`.
- **`REGISTRY` is the list of what the format promises to read back**, keyed by
  `module:qualname`: every fit, every result read off one, their specs and parts, the panel, the
  tests and the causal-hybrid-control classes a fit holds or a plan's goal names. `save` refuses a
  value of any other class and names where it sits; `load` refuses a file that names one. It leaves
  out `Refit`, which holds a function, `Schema`, which holds a type, the sampler's and the solvers'
  working classes and the compiled models: `tests/test_roundtrip.py` names each with its reason, and
  fails on a public class that is in neither list.
- **A loaded object is rebuilt through its own `__init__`**, so it passes the checks a fitted one
  passed, and it equals the saved one to the bit: every array's dtype, shape and bytes, every float,
  every mapping's kind and order. A JAX array loads as one; a file holding one in double precision
  is refused where JAX runs in single, which would round it.
- **The version rule.** `FORMAT_VERSION` is 1. It moves when a file the current version wrote
  would load differently, or not at all: a registered class or a field renamed, moved or removed, a
  field added without a default, or a field's meaning or units changed. Adding a class to the
  registry, or a field with a default, does not move it. `load` reads its own version only and
  refuses any other, naming the file's version, the mixdyn that wrote it and the one that reads
  it, never reading it as its own. When the version moves, the release that moves it reads the
  previous version too, through a converter, for at least one minor release.
- **The format is the engine's, for Python callers.** Callers outside Python read the versioned
  JSON documents of `mixdyn.api` (`ISSUES`), which state their schema field by field.

### The public surface

- **A name is public when an `__all__` of a package lists it**: `mixdyn`, `mixdyn.api`,
  `mixdyn.calibration`, `mixdyn.data`, `mixdyn.decisions`, `mixdyn.experiments`,
  `mixdyn.inference`, `mixdyn.models` and `mixdyn.transforms`, and the `__all__` of
  `mixdyn.api.serialize`. So are the fields, methods and properties of the classes they name that
  do not start with an underscore. A name reachable only by its module path is not public yet,
  however much it is used; one that should be joins an `__all__`. *Amended 2026-11-25 (ADR 0102):*
  every module's `__all__` is public, not the packages' and `mixdyn.api.serialize`'s alone:
  `mixdyn.data.feedback`, `mixdyn.models.demand` and `mixdyn.inference.sensitivity` have one, and
  the test and the index read them already.
- **`tests/test_public_names.py` holds the surface to its documentation.** Every name an
  `__all__` lists imports, none is listed twice or is private, and each is documented: a docstring
  of its own, the string after a constant or alias in the module that defines it, or a page that
  cites it. The names not yet documented are listed there and the list only shrinks, as
  `tests/test_self_contained.py` treats pointers. Every `mixdyn.` name a page cites must import.
  Once `docs/api/index.md` exists, it lists every public name. *Amended 2026-11-06:* it exists,
  written by `scripts/api_index.py` from the `__all__`s and each name's first sentence, with this
  ADR's rules for versions, deprecation and the format. `tests/test_public_names.py` fails when
  the page is not the one the script writes. A name the page lists is not documented by it: the
  page cites every name, so only the other pages count. *Amended 2026-11-25 (ADR 0103):* every
  public name is documented, and the list of names not documented yet is gone.
- **Versions.** Before 1.0, a minor release may change the public surface, and only as the rule
  below allows; a patch release never removes or changes a public name. From 1.0, a name leaves
  only in a major release. *Amended 2026-11-25 (ADR 0103):* a public name's signature, fields and
  members change by kind. A function or a method may gain parameters with defaults in any release;
  a default or a value moves only in a minor release; any other change comes only in a major
  release from 1.0, and before 1.0 not at all: the name is renamed instead.
- **How a public name is removed:** for one minor release it still imports, and using it raises a
  `DeprecationWarning` that names its replacement and the release it leaves in; it leaves in the
  minor release after, and the changelog names it under *Removed*. causal-hybrid-control removes
  its names the same way. A renamed class stays in `REGISTRY` under its old key for as long as its
  old name imports, so files that name it still load. *Amended 2026-11-25 (ADR 0102):* the old
  name leaves its package's `__all__` for the package's `_DEPRECATED`, which names its replacement
  and its leaving release, and imports through the package's `__getattr__`;
  `tests/test_public_api.py` holds each to this rule, and the snapshot it reads,
  `tests/public_api.txt`, is not written again while it would drop a name before its release.

### The reader's contracts

- **Channels are read by name.** The order of the data's columns moves nothing; the panel holds
  the channels in `spend`'s order, each with its own column, and an engine reads them by name, so
  permuting `spend` leaves every return as it was.
- **The dates are spaced as the cadence says.** Where most rows miss the cadence's grid,
  `read_panel` names the spacing it found and refuses it: data finer than the cadence is to be
  resampled before reading, which the reader does not do, and data coarser than the cadence is to
  be read at its own.
- **A geo's population is one number** over the calendar: given once a geo, or read from a column
  whose rows agree within each geo. A series, or a column that changes within a geo, is refused.
- **A geo observing the KPI in fewer than 15 periods per channel and control warns**
  (`IssueCode.FEW_OBSERVATIONS`): its effects are barely identified, and their intervals lean on
  their priors. The warning does not stop a fit.

## Consequences

- A fit or a result outlives the process that drew it, and a file is read back the same to the bit
  or refused, never read as something else. `tests/test_roundtrip.py` fits every engine on a tiny
  world, saves and loads it, and holds every return, every interval's ends, every plan and every
  forecast read off the loaded fit to those read off the fitted one, exactly.
- A registered class's fields are now part of the format: renaming one is a format change, and
  moves `FORMAT_VERSION` unless it keeps an alias.
- The engines' entry points that sit at module paths (`fit_ridge`, `fit_bayes`, `fit_bayes_geo`,
  `fit_nested`, their specs and fits) are not public until an `__all__` lists them. *Amended
  2026-11-25 (ADR 0102):* `mixdyn.models` lists them, and every class a public name takes, returns
  or holds is public too.
- A panel too short for its effects carries a warning in `issues` that it did not carry before.
