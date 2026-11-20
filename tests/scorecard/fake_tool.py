"""A tool's script for the harness's tests, run as the bench's are: `--worlds DIR --out DIR`, every
world of `DIR` that `--out` does not hold fitted in turn, one JSON record a world. Each world ends
as `--modes` names it for its seed:

* `record`: records the status quo's plan;
* `raise`: records the error a fit raised, as the bench's scripts record a world that breaks them;
* `exit`: exits with code 3 and no record;
* `memory`: allocates until its scope's memory cap kills it;
* `segfault`: dies by SIGSEGV;
* `half`: in its first attempt writes half its record and dies by SIGKILL, then records;
* `sleep`: sleeps past any timeout.

Each start is counted in `<modes>.starts/<seed>`.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import signal
import sys
import time
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worlds", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--modes", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))  # a segfault leaves no core
    modes = json.loads(args.modes.read_text())
    starts = args.modes.with_suffix(".starts")
    starts.mkdir(exist_ok=True)
    for path in sorted(args.worlds.glob("world_*.npz")):
        target = args.out / f"{path.stem}.json"
        if target.exists():
            continue
        with np.load(path) as saved:
            seed = int(saved["seed"])
            record = {
                "version": int(saved["version"]),
                "seed": seed,
                "digest": str(saved["digest"]),
                "error": None,
                "weekly": saved["status_quo"].tolist(),
            }
        with (starts / str(seed)).open("a") as count:
            count.write("started\n")
        started = len((starts / str(seed)).read_text().splitlines())
        mode = modes[str(seed)]
        if mode == "raise":
            record |= {"error": "RuntimeError: the fit broke", "traceback": "Traceback"}
        elif mode == "exit":
            sys.exit(3)
        elif mode == "memory":
            held = []
            while True:
                held.append(np.ones(2**20))
        elif mode == "segfault":
            os.kill(os.getpid(), signal.SIGSEGV)
        elif mode == "half" and started == 1:
            text = json.dumps(record)
            target.write_text(text[: len(text) // 2])
            os.kill(os.getpid(), signal.SIGKILL)
        elif mode == "sleep":
            time.sleep(3600)
        target.write_text(json.dumps(record))


if __name__ == "__main__":
    main()
