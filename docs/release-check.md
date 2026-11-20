# Release check: the wheel and the sdist, built from a clean clone

A record of one build, made before the public repository exists, of what the package would ship
(ADR 0054). It is redone on the public repository's first tag, where `.github/workflows/release.yml`
repeats the version, licence and classifier checks on every release.

## How it was built

On 2026-11-06, from commit `a62e9a4`, whose `src/` is `9c36b9e`'s:

```bash
git clone --no-local <mixdyn checkout> <scratch>/clone
cd <scratch>/clone && uv build --offline --out-dir <scratch>/dist
uv venv <scratch>/wheel-venv --python 3.14
uv pip install --offline --python <scratch>/wheel-venv/bin/python \
    <scratch>/dist/mixdyn-0.0.1-py3-none-any.whl
<scratch>/wheel-venv/bin/python -I import_all.py
```

uv 0.13.7, hatchling from uv's cache; `--offline` resolved every package from that cache, so the
versions below are the newest the cache held, not the lock's. `import_all.py` imports `mixdyn`
and every module under it, and reads every name of every `__all__` from its module. `-I` keeps the
working directory and `PYTHONPATH` off the import path, so only the installed wheel is read.

## The wheel

- **`mixdyn-0.0.1-py3-none-any.whl`, 264037 bytes**, 56 files: 50 modules and `py.typed` under
  `mixdyn/`, and 5 under `mixdyn-0.0.1.dist-info/` (`METADATA`, `WHEEL`, `RECORD`, and
  `licenses/LICENSE` and `licenses/NOTICE`).
- **No file outside the package:** no `tests/`, `evidence/`, `scorecard/`, `validation/`,
  `scripts/` or `docs/` path. hatchling builds the wheel from `src/mixdyn` alone
  (`[tool.hatch.build.targets.wheel]`), and `tests/test_wheel_contents.py` fails when the build's
  configuration would let another path in, without building.
- **The metadata:** `Metadata-Version: 2.5`; `License-Expression: Apache-2.0`; `License-File:
  LICENSE` and `License-File: NOTICE`; `Requires-Python: >=3.12`; `Requires-Dist:
  causal-hybrid-control>=0.22.0`, `numpy>=2.3`, `scipy>=1.16`; the classifiers `Private :: Do Not
  Upload` and `Typing :: Typed`. The private classifier stays until the publication: PyPI refuses
  it, and the release workflow stops on it.

## The import

In a fresh environment of Python 3.14.8 holding the wheel and what it requires
(causal-hybrid-control 1.1.0, jax 0.12.0.1, numpy 2.5.4, scipy 1.18.1, and theirs), `mixdyn` was
read from the environment's `site-packages`; 50 modules imported, 11 of them with an `__all__`, and
each of their 201 entries (184 distinct names) was found. Nothing failed.

What this does not show:

- **Only the import ran, not a fit.** The resolver took causal-hybrid-control 1.1.0, the newest
  the cache held; the lock holds 0.22.0 and the suite runs on it. The requirement's floor admits
  every later release.
- **jax is not declared.** Ten modules import it, and it arrives as a requirement of
  causal-hybrid-control.

## The sdist

**`mixdyn-0.0.1.tar.gz`, 3251321 bytes, 299 files**: hatchling's default, every tracked file of
the tree (`src/`, `tests/`, `docs/`, `validation/`, `evidence/`, `scorecard/`, `scripts/`,
`.github/`, `uv.lock` and the rest). The records under `evidence/` carry the local paths of the
checkouts their runs read (`evidence/README.md`), and the sdist ships them as they are.

## Redone for 0.1.0, 2026-11-20

Built with uv 0.13.9 (`uv build`) from the tracked files of the release commit, the tree the public
repository's first commit holds:

- **The wheel, `mixdyn-0.1.0-py3-none-any.whl`, 366450 bytes, 64 files:** 58 modules and
  `py.typed` under `mixdyn/`, and the 5 files of `mixdyn-0.1.0.dist-info/`. No path outside the
  package.
- **Its metadata:** `Metadata-Version: 2.5`; `License-Expression: Apache-2.0` with `LICENSE` and
  `NOTICE`; no `Private :: Do Not Upload` and no licence classifier (ADR 0095); `Requires-Python:
  >=3.12`; `Requires-Dist: causal-hybrid-control>=1.2.0`, `jax>=0.4.30`, `numpy>=2.3`,
  `scipy>=1.16`. jax, which the build of 2026-11-06 found undeclared, is declared.
  `twine check --strict` passes on the wheel and the sdist, and the release workflow's own check
  of the version, the licence and the classifier passes on the wheel.
- **The sdist, `mixdyn-0.1.0.tar.gz`, 218 files:** the package, `docs/`, `scripts/`,
  `validation/`, `.github/`, the lock and the project's files. It leaves out `evidence/`,
  `scorecard/` and `tests/`, which run from the repository: the tests read the other two.
- **The install:** in a fresh Python 3.14 environment holding the wheel and what it requires
  (causal-hybrid-control 1.3.1, jax 0.12.1, numpy 2.5.5, scipy 1.18.1), `mixdyn` was read from the
  environment's `site-packages` and the README's example ran, printing the plan it prints from the
  repository. With causal-hybrid-control at its floor, 1.2.0, the example printed the same.
