# 0095. Public on PyPI from a fresh history

Status: accepted, 2026-11-20; supersedes ADR 0001's last decision, installation from the git
source alone, and ADR 0054's rule that the private classifier stays until the public repository
exists

## Context

ADR 0001 kept the engine private: it installed from its git source, never from an index, and the
`Private :: Do Not Upload` classifier made PyPI refuse an accidental upload. ADR 0054 settled what
publication needs: the Apache-2.0 licence, contributions under the Developer Certificate of
Origin, Python 3.12 and later, a release workflow that publishes from a tag, and a public
repository that starts from a fresh history of the cleaned tree, never this repository's.

It left three things for the first public commit: the classifier removed, a `CHANGELOG.md` with
the first version's section, and the trusted publisher on PyPI.

The launch waited for a pre-registered scorecard run that measures the engine against the tools it
is compared with, so that it carries a number. That run, M3, read its claim on 2026-11-20: 17 of
17 verdicts held, with losses beside them (README). It did not cover every family of the bench:
families 11, 16 and 21 were not in it, and no headline was readable. On 2026-11-20 the author
decided to publish on M3's result, with those gaps stated.

## Decision

- **The engine is published on PyPI as `mixdyn`, version 0.1.0.** `pyproject.toml` drops the
  `Private :: Do Not Upload` classifier and states the development status, the audience, the
  Python versions and the project's URLs. ADR 0001's last decision is superseded: the engine
  installs from the index. Its allow-list of runtime packages, checked against the lock, stands.
- **No licence classifier.** The licence is the PEP 639 expression `Apache-2.0` with `LICENSE` and
  `NOTICE` as licence files (ADR 0054). PEP 639 deprecates the `License ::` classifiers beside a
  licence expression and lets tools refuse the pair, so the metadata states the licence once.
- **The public repository starts from one commit of this tree**, made by `git archive` of the
  release commit: the tracked files, nothing of this repository's history. This repository is
  never made public.
- **Commit hashes written before 0.1.0 are kept as written.** ADRs, `evidence/` and
  `docs/release-check.md` name commits of the private history. They identify what ran, and in the
  public repository they resolve nowhere. Where a published claim rests on a commit, the README
  names the content instead: the git tree hashes of `src/` and `scorecard/` that M3 declared, which
  the public repository's first commit carries unchanged.
- **The records in `evidence/` lose their local paths.** Each record's provenance named the local
  directory of each checkout it read. Those paths are replaced by `<local checkout>` and, for an
  installed module, `<venv>`; the commits, the dirty flags and every number stay as the run wrote
  them.
- **ADR 0054's list of what stays private is reworded**, and so is `evidence/README.md`'s; what
  stays private is unchanged.
- **Releases come from signed tags** in the public repository, through
  `.github/workflows/release.yml` as ADR 0054 drafted it: PyPI Trusted Publishing (OIDC) from the `pypi` environment, no stored
  token, the release notes the version's section of `CHANGELOG.md`.

## Consequences

- `pip install mixdyn` installs the engine and its runtime closure from PyPI.
- The name on the index is the project's own from its first upload, so the risk ADR 0001 named, a
  public package of the same name winning the resolution, no longer applies.
- A reader of the public repository cannot follow a commit an ADR or a record names, and cannot see
  the history behind a decision; the ADRs state the decisions themselves.
- M3's pre-registration has no public time-stamp before the run: it was frozen in this private
  repository. The README says so beside the claim.
- The private repository goes on as the author's working copy; a change reaches the public one as
  a commit there.

## Alternatives considered

- **Wait for a run over every family of the bench.** Rejected by the author on 2026-11-20: the
  launch carries M3's claim and states the families it did not cover.
- **Make this repository public.** Rejected, as in ADR 0054: its history holds work that stays
  private.
- **Keep the `License :: OSI Approved :: Apache Software License` classifier beside the
  expression.** Rejected: PEP 639 deprecates it, and the expression already states the licence.
- **Rewrite the old commit hashes.** Rejected: a hash identifies what ran only in the history it
  belongs to, and replacing it with a public commit would claim a run that never happened there.
