# mixdyn

--8<-- "README.md:intro"

## How it measures up

![mixdyn against its closest rival, family by family](assets/closest-rival-light.svg#gh-light-mode-only)
![mixdyn against its closest rival, family by family](assets/closest-rival-dark.svg#gh-dark-mode-only)

--8<-- "README.md:measured"

## Install

--8<-- "README.md:install"

## A first plan

--8<-- "README.md:example"

![The first plan, drawn](assets/first-plan-light.svg#gh-light-mode-only)
![The first plan, drawn](assets/first-plan-dark.svg#gh-dark-mode-only)

--8<-- "README.md:example-notes"

## Where it sits

Marketers' measurement is turning to MMM checked by experiments, more than to
multi-touch attribution (MTA), which follows each user's path to a sale. In a survey of 196 US
marketers that EMARKETER fielded for TransUnion in July 2025, 47 % planned to raise their spend on
MMM in the next year and 35 % their spend on MTA
([TransUnion, October 2025](https://newsroom.transunion.com/new-transunion-research-reveals-marketers-confidence-in-measurement-has-stalled/)).
Forrester's 2025 and 2026 global marketing surveys found that "adoption of marketing mix
modeling and incrementality testing increased substantially year over year"
([Forrester, July 2026](https://www.forrester.com/blogs/ai-moves-marketing-measurement-from-insights-to-action/)).
CIMM's 2026 roadmap for MMM asks the industry to "establish an experimentation and validation
playbook for geo tests, lift tests"
([CIMM, July 2026](https://cimm-us.org/cimm-outlines-industry-roadmap-to-strengthen-marketing-mix-modeling-as-ai-accelerates-media-decision-making/)).
mixdyn is built for that pairing: its Bayesian models read lift and geo tests in the likelihood,
and `mixdyn.experiments` prices a test by what it adds to the plan before it runs (ADRs 0030 to
0032). It does not do MTA: it reads no user-level path.

## Using it

Each channel's return can be read against an unmeasured confounder: `mixdyn.inference.sensitivity`
gives Cinelli and Hazlett's robustness values for the ridge engine and the marginal sensitivity
model's `Gamma*` for the Bayesian ones, `mixdyn.api.sensitivity_line` states them, and
`mixdyn.api.factsheet` writes each Bayesian channel's line among its limitations (ADR 0061). The
bound is wide wherever the spend moved little beside the KPI's unexplained variation, confounded
or not.

A static fit can be checked on the periods after its window and carried forward as weeks arrive
(ADR 0079). `holdout_bayes` scores the posterior predictive on the window's last periods, held out
of the fit: periods drawn from the whole window would share their autocorrelated noise with their
fitted neighbours and flatter the score. `refresh_bayes` refits the whole panel with each
channel's return prior tempered by the previous posterior, and reports how far each return moved
and whether it left the previous 80 % interval. The scorecard's refresh cell
(`python -m scorecard.refresh`) refits every arm on a world's history four weeks at a time, six
times, and reads how far each return and each plan's regret swing between refits, and, from a
rolling origin on the short-history family, each origin's forecast score. Measurement services
sell a refresh loop whose results stay stable; the cell measures that stability on worlds whose
truth is known, for this engine's arms and the rivals'.

## Where to go next

- [Benchmark](benchmark.md): what three pre-registered runs measured against Meridian,
  PyMC-Marketing, Orbit, the status quo and the equal split, where the engine loses, and how each
  run can be reproduced.
- [Walkthrough](walkthrough.md): one panel through read, fit, diagnostics, tests, plan and report,
  each step gated by the engine's own checks.
- [Scope](scope.md): what the engine does not model, reach and frequency among it.
- [API reference](api/index.md): every public name, the versioning rules, and one page per public
  module rendered from its docstrings.
- On GitHub: the [decision records](https://github.com/causaldyn/mixdyn/tree/main/docs/adr) the
  code relies on, the [evidence](https://github.com/causaldyn/mixdyn/tree/main/evidence) their
  measurements come from, the [scorecard](https://github.com/causaldyn/mixdyn/tree/main/scorecard)
  harness, the [changelog](https://github.com/causaldyn/mixdyn/blob/main/CHANGELOG.md) and how to
  [contribute](https://github.com/causaldyn/mixdyn/blob/main/CONTRIBUTING.md).
