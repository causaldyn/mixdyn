# Contributing

Issues and pull requests are welcome. Please open an issue before a large change, so that two people
do not build the same thing differently.

## Setup

```bash
just sync    # .venv exactly as uv.lock has it
just check   # format, lint, types, the dependency gate, tests: what CI runs
```

Python 3.12 and up; development runs on 3.14 (`.python-version`). `uv.lock` is committed and
authoritative: run everything through `uv run`, never a bare `python`, `pytest` or `ruff`, or you
are testing a different dependency set from CI's.

## The gates

CI runs these, and a pull request is expected to pass them locally first:

```bash
uv run ruff format --check .
uv run ruff check .
uv run ty check                            # at Python 3.12, the lowest supported version
uv run python scripts/check_runtime_deps.py
uv run pytest
```

The full suite samples several posteriors and takes a while; CI splits it into groups
(`.github/workflows/ci.yml`) and runs the rest of it on Python 3.12, 3.13, 3.14 and 3.15.

## What a change should look like

- **Runtime packages are an allow-list** (`allowed-runtime-packages.txt`, ADR 0001). A new one needs
  an issue first. Special-purpose packages (a probabilistic-programming language, an MMM library, a
  statistics framework) are used as oracles in a scratch environment, never in the lock.
- **Every numerical routine has an independent oracle**: a closed form, another implementation,
  coverage over seeds, or simulation-based calibration. Say which in the test's docstring.
- **New behaviour comes with its tests.** A pull request that adds or changes behaviour adds the
  tests that show it, in the same pull request; a bug fix adds a test that fails without the fix.
- **Tests are two-sided.** A test that only shows the good case passing does not show that the bad
  case fails; prefer one that fails when the implementation is reverted.
- **No number in a docstring, the README or the changelog that a test or a script does not
  produce.**
- **Scope claims precisely.** If a result holds only for one geo, a known variance or a linear
  response, the docstring says so.
- **A claim that mixdyn beats another tool** names its axis, its family of worlds and its paired
  interval from a pre-registered scorecard run, and states the losses beside it.
- A decision the code will rely on gets an ADR in `docs/adr/`: context, decision, consequences,
  alternatives.
- Everything written, code, comments, docstrings, commit messages and issues, is in English.
- Commit messages use a conventional-commit prefix (`feat:`, `fix:`, `refactor:`, `test:`,
  `docs:`, `chore:`) and explain the *why* in the body.

## Sign-off

Contributions are accepted under the [Developer Certificate of Origin](https://developercertificate.org/)
(DCO) 1.1: by signing off a commit you certify that you wrote it, or otherwise have the right to
submit it under the project's licence. Sign off each commit with `git commit -s`, which adds

```
Signed-off-by: Your Name <you@example.org>
```

A commit without a sign-off is not merged. There is no contributor licence agreement.
Contributions are licensed under the project's licence, Apache-2.0 (ADRs 0054 and 0095).
