# 0054. The licence and the road to publication

Status: accepted, 2026-10-29; amended 2026-11-06: the harnesses the ADRs cite live in `evidence/`,
CI runs the suite on Python 3.12 to 3.15, a release workflow is drafted, and the name `mixdyn` was
free on PyPI; amended 2026-11-20 by ADR 0095: the classifier leaves with the release commit of
0.1.0, the public repository's first commit, and the list of what stays private is reworded

## Context

mixdyn is to be published. It will not be published by opening this repository: its history
carries work that stays private, so the public repository starts from a fresh history of a cleaned
tree. Three things can be settled before then: the licence the tree carries, how contributions are
accepted, and what stays out of the public tree.

The engine is built on causal-hybrid-control, which is Apache-2.0. The packages it is compared
with are permissive too:

| package | licence |
| --- | --- |
| Meridian (Google) | Apache-2.0 |
| PyMC-Marketing | Apache-2.0 |
| LightweightMMM (Google) | Apache-2.0 |
| Robyn (Meta) | MIT |

A user who already runs one of them can take mixdyn on the same terms, and code can move between
mixdyn and causal-hybrid-control without a second licence to reconcile.

## Decision

- **The licence is Apache-2.0.**
  - `LICENSE` is the Apache License 2.0 text, byte for byte as apache.org and
    causal-hybrid-control's distribution carry it.
  - `NOTICE` names the project and its copyright holder; section 4(d) of the licence makes every
    redistribution carry it.
  - `pyproject.toml` states it by PEP 639: `license = "Apache-2.0"` and
    `license-files = ["LICENSE", "NOTICE"]`, so a wheel's metadata reads `License-Expression:
    Apache-2.0` and the wheel and the sdist carry both files.
- **Why Apache-2.0 and not MIT or BSD:** its express patent grant (section 3). Every contributor
  licenses the patents their contribution reads on, and the grant ends for anyone who sues over
  them. MIT and BSD say nothing about patents. A model, a sampler or an optimiser can read on a
  patent in a way a utility library rarely does.
- **Contributions are accepted under the Developer Certificate of Origin 1.1**, with no
  contributor licence agreement. Each commit carries a `Signed-off-by:` line (`git commit -s`):
  its author certifies they have the right to submit it under the project's licence. Apache-2.0's
  section 5 already licenses a contribution on the project's terms; the sign-off records that the
  contributor was entitled to give it. A CLA would add a signature process and assign rights the
  project does not need.
- **What stays private:**
  - a private package outside this tree, which will not be part of it;
  - the research harnesses some ADRs cite as the place a measurement was run. Until they have a
    public home, `tests/test_self_contained.py` lists every line that points at them and fails on
    any new one, so the list only shrinks. *Amended 2026-11-06:* the harnesses those lines cited,
    the scripts they import and the records their numbers come from moved into `evidence/`
    (`evidence/README.md`), the ADRs name those files, and the list is empty. The R scripts that
    recorded the R oracles' fixtures, and that private package's harness, stay private.
- **ADR 0001's installation policy holds until the public repository exists.** The engine still
  installs from its git source, and the `Private :: Do Not Upload` classifier stays: it leaves only
  in the public repository's first commit. The licence does not publish anything by itself.
- **The engine supports Python 3.12 and later.** `requires-python = ">=3.12"`, ruff's target and
  ty's environment are 3.12, the lowest supported version, and development stays on 3.14
  (`.python-version`). A public package that only the newest CPython can install excludes most of
  its readers. *Amended 2026-11-06:* CI runs the suite without the posterior groups on 3.12 and
  3.13 beside 3.14 and 3.15; `uv.lock` resolves on each, every compiled package in it carrying
  wheels for all four.
- **Publication from a tag** (amended 2026-11-06). `.github/workflows/release.yml` is drafted for
  the public repository: on a `v*` tag it builds with uv on Python 3.12, checks that the wheel
  carries the tag's version, `License-Expression: Apache-2.0` and not the private classifier,
  builds the release notes from the version's section of `CHANGELOG.md`, publishes through PyPI
  Trusted Publishing (OIDC, the `pypi` environment, `pypa/gh-action-pypi-publish` pinned to a
  commit), then creates the GitHub Release. It never runs here: no tag is pushed to this
  repository, and the classifier stops it before an upload.
- **The name on PyPI** (amended 2026-11-06). `https://pypi.org/pypi/mixdyn/json` and the simple
  index's `mixdyn` page answered 404 on 2026-11-06 at 19:01 UTC: no project holds the name. A 404
  reserves nothing, and PyPI can still refuse a name at the first upload if it reads it as
  confusable with an existing project; the name is settled only by that upload.

## Consequences

- Every file in the tree is under Apache-2.0 from the commit that adds `LICENSE`. The private
  repository's history is not published, so nothing before that commit needs relicensing.
- A contribution without a sign-off cannot be merged in the public repository; the private one has
  a single author and needs no check.
- Code that only a newer CPython accepts (PEP 758's unparenthesised `except A, B:`,
  `itertools.batched(strict=)`) fails ruff and ty at their 3.12 target.
- Before the first public commit: the ADRs' pointers to the research harnesses are resolved (each
  measurement's harness published, or the pointer replaced by the result it records), and the
  classifier is removed. *Amended 2026-11-06:* the pointers are resolved; the classifier, a
  `CHANGELOG.md` with the first version's section, and the trusted publisher on PyPI remain.

## Alternatives considered

- **MIT.** Rejected: no patent grant, for a codebase whose methods can read on patents.
- **A copyleft licence (GPL-3.0, AGPL-3.0).** Rejected: the packages users compare it with are
  permissive, and causal-hybrid-control, which it builds on, is Apache-2.0.
- **A CLA.** Rejected: the DCO records what the project needs, the contributor's right to submit,
  without a separate agreement.
- **Opening this repository as it is.** Rejected: its history carries the private work above. A
  fresh history of the cleaned tree publishes only what is meant to be public.
