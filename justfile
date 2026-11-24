# Verification loop. `just check` runs what ci.yml runs, in the same order, so a green check here
# is a green CI. CI invokes the commands directly; if a recipe and ci.yml ever disagree, ci.yml is
# the one that counts.

default:
    @just --list

# Everything, cheapest first. Stops at the first failure.
check: fmt lint types deps test

# `.venv` exactly as the lock has it. The interpreter is the build with the GIL, the one CI runs:
# a bare "3.14" request can settle on the free-threaded build here.
sync:
    UV_PYTHON=3.14+gil uv sync --locked

fmt:
    uv run ruff format --check .

lint:
    uv run ruff check .

types:
    uv run ty check

# The runtime packages uv.lock resolves must all be on allowed-runtime-packages.txt (ADR 0001).
deps:
    uv run python scripts/check_runtime_deps.py

test:
    uv run pytest

# One fuzz target, `documents`, `plan` or `read`, for `seconds`, as fuzz.yml runs it; its corpus
# stays in .fuzz/, and a crash writes the input that caused it there.
fuzz target="read" seconds="60":
    mkdir -p .fuzz/{{target}}
    uv run --with-requirements fuzz/requirements.txt python fuzz/fuzz_{{target}}.py .fuzz/{{target}} -max_total_time={{seconds}} -max_len=1024 -len_control=0 -artifact_prefix=.fuzz/

fix:
    uv run ruff check --fix .
    uv run ruff format .
