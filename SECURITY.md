# Security policy

## Supported versions

| Version | Supported |
|---|---|
| 0.1.x | yes |

This is a single-author engine. Only the latest minor gets fixes; there are no backports. The
upgrade path is the [changelog](https://github.com/causaldyn/mixdyn/blob/main/CHANGELOG.md), which
records every behaviour change, not only the additions.

## Reporting a vulnerability

Use GitHub's [private vulnerability reporting](https://github.com/causaldyn/mixdyn/security/advisories/new).
It is enabled on this repository, so the report stays private until a fix ships. Do not open a public
issue for a vulnerability.

Expect a first response within **7 days**. That is a single maintainer's honest number, not a
service level: if it matters to you commercially, say so in the report and say what you need.

Useful in a report: the version, the Python version, the minimal input that triggers it, and what an
attacker gains. A proof of concept is welcome; a CVSS score is not required.

## What is in scope

- **In scope.** Code execution or file access reachable from data a caller passes in: a panel or
  frame given to `mixdyn.data`, a JSON document read by `mixdyn.api.documents`, a saved model archive
  loaded by `mixdyn.api.serialize`. The archive format holds JSON and `.npy` arrays read with
  `allow_pickle=False`; any path that makes loading one run code is a vulnerability. A dependency
  with a known advisory that this package pins into place is in scope too.
- **In scope, and treated as a bug of the same severity.** Any path where the engine writes a
  secret, a token or a credential into a log record or an exception message.
- **Out of scope.** Wrong numbers: a miscalibrated interval, a plan that loses money, an estimator
  biased under an assumption it documents. These are correctness bugs, and they matter more to this
  project than most security bugs do, but they go in the public issue tracker where they can be
  argued about in the open.
- **Out of scope.** Denial of service by giving the engine a large problem. It uses the memory the
  problem needs.

## Supply chain

- Releases are published to PyPI by **Trusted Publishing** (OIDC), from `.github/workflows/release.yml`
  on a signed `v*` tag, after a reviewer approves the `pypi` environment. No API token exists to leak.
- Every file on PyPI carries a **PEP 740 attestation**, signed through Sigstore, that it was uploaded
  by this workflow. Check it before you trust a file:

  ```bash
  curl -s https://pypi.org/integrity/mixdyn/<version>/mixdyn-<version>-py3-none-any.whl/provenance
  ```

  The publisher in a valid attestation is `causaldyn/mixdyn` via `release.yml` and the environment
  `pypi`. A wheel for this project that carries no attestation, or one naming a different workflow,
  did not come from here.
- `uv.lock` is committed and authoritative. The runtime packages are an allow-list
  (`allowed-runtime-packages.txt`) that CI checks the lock against.
