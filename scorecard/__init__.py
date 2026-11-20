"""The scorecard's harness: this engine's arms run on a family of worlds whose truth is known, and
every arm's plan scored the same way.

The worlds, their truth, the scoring and the gate are the bench's (`causaldyn_bench.scorecard`),
which enters through `PYTHONPATH` and is never a dependency of the engine. A run lives in one
directory (`scorecard.layout`): it is declared once (`run.json`), each world is exported once as
the bench archives it, and each arm, this engine's (`scorecard.arms`) or a tool's run by the
bench's own scripts (`scorecard.tools`), leaves one record a world. `scorecard.run` exports, runs
every arm by one rule, each world in a process of its own, scores every record against the
world's truth and reports; it resumes wherever a file is missing.

The harness ships with the engine's sources, not in its wheel.
"""
